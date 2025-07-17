# modified from: https://github.com/xuweijieshuai/Neural-Topic-Modeling-vmf/tree/main/src/topicmodeling
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from preprocess.preprocess import TextProcessor
from spherical_distribution.distributions.hyperspherical_uniform import HypersphericalUniform
from spherical_distribution.distributions.von_mises_fisher import VonMisesFisher
import torch.nn.functional as F

def get_mlp(features, activate):
    if isinstance(activate, str):
        activate = getattr(nn, activate)
    layers = []
    for in_f, out_f in zip(features[:-1], features[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activate())
    return nn.Sequential(*layers)

def topic_covariance_penalty(topic_emb, EPS=1e-12):
    #normalized the topic
    normalized_topic = topic_emb / (torch.norm(topic_emb, dim=-1, keepdim=True) + EPS)
    #get topic similarity absolute value
    cosine = (normalized_topic @ normalized_topic.transpose(0, 1)).abs()
    #average similarity
    mean = cosine.mean()
    #variance
    var = ((cosine - mean) ** 2).mean()
    return mean - var, var, mean

class VNTM(nn.Module):
    def __init__(self, hidden, h_to_z, topics, layer, top_number, penalty, lambda_sh, kappa, gamma, temp=10):
        super(VNTM, self).__init__()
        self.hidden = hidden
        self.h_to_z = h_to_z
        self.decoder = topics
        self.output = None
        self.drop = nn.Dropout(p=0.5)
        self.fc_mean = nn.Linear(layer, top_number)
        self.fc_var = nn.Linear(layer, 1)
        self.num = top_number
        self.penalty = penalty
        self.temp = temp
        self.lambda_sh = lambda_sh
        self.kappa = kappa
        self.gamma = gamma

    def forward(self, x, device, n_sample=1):
        h = self.hidden(x)
        h = self.drop(h)

        z_mean = self.fc_mean(h)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)

        z_var = F.softplus(self.fc_var(h)) + self.kappa

        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(self.num - 1, device=device)
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean().to(device)

        rec_loss = 0
        log_prob = 0
        vEC_loss = 0

        for i in range(n_sample):

            z = q_z.rsample()
            z = self.h_to_z(self.temp * z)
            self.output = z
            log_prob, vEC_loss = self.decoder(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)

        rec_loss = rec_loss / n_sample
        vEC_loss = vEC_loss/ n_sample

        minus_elbo = rec_loss + kld

        penalty, var, mean = topic_covariance_penalty(self.decoder.mu_mat)

        return {
            'loss': minus_elbo + self.penalty * penalty + self.gamma * vEC_loss,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld,
            'penalty': penalty,
            'log_prob': log_prob,
            'vEC_loss': vEC_loss
        }

    def get_topics(self):
        return self.decoder.get_topics()

class Generative_model(nn.Module):
    def __init__(self, word_embedding, kappa, n_topics=50, topic_embeddings_size=100, temp=10):
        super(Generative_model, self).__init__()

        self.device = torch.device('cuda')
        self.kappa2 = kappa

        self.n_topics = n_topics
        self.topic_embeddings_size = topic_embeddings_size
        self.temp = temp

        self.theta_softmax = nn.Softmax(dim=1)
        self.theta_linear = nn.Linear(n_topics, n_topics)

        self.theta_linear.bias.data.fill_(0.0)
        self.theta_linear.weight.data.fill_(0.0)

        self.beta_softmax = nn.Softmax(dim=1)

        word_embeddings_mat = word_embedding.weight.clone().detach()
        # torch.nn.init.xavier_uniform_(word_embeddings_mat)
        word_embeddings_mat = F.normalize(word_embeddings_mat, p=2, dim=1)
        word_embeddings_mat = nn.Parameter(word_embeddings_mat.requires_grad_(True))
        self.register_parameter('word_embeddings_mat', word_embeddings_mat)

        mu_mat = Parameter(torch.Tensor(n_topics, topic_embeddings_size))
        torch.nn.init.xavier_normal_(mu_mat.data, gain=1)
        mu_mat.requires_grad = True
        self.register_parameter('mu_mat', mu_mat)

        kappa = Parameter(torch.Tensor(n_topics))
        kappa.data.fill_(self.kappa2)
        kappa.requires_grad = True
        self.register_parameter('kappa', kappa)

    def log_vmf_pdf(self, mu, kappa, vec):
        mu = mu / torch.norm(mu, dim=1, keepdim=True)
        vec = vec / torch.norm(vec, dim=1, keepdim=True)

        cosine_similarity = torch.matmul(mu, vec.T)

        d = self.topic_embeddings_size
        log_c = (d / 2 - 1) * torch.log(kappa + 1e-10) - (d / 2) * torch.log(torch.tensor(2 * torch.pi)) - (kappa + 1e-10)

        small_kappa = kappa < 1e-2
        if small_kappa.any():
            log_c[small_kappa] = (d / 2 - 1) * torch.log(kappa[small_kappa] + 1e-10) - torch.log(torch.tensor(2 * torch.pi)) - torch.log(torch.exp(-kappa[small_kappa]) + 1e-10)

        log_prob = kappa.view(-1, 1) * cosine_similarity + log_c.view(-1, 1)
        return log_prob

    def vEC_loss(self, word_embeddings, topic_mu, topic_kappa, epsilon=1e-10, sinkhorn_iter=50):
        word_embeddings = F.normalize(word_embeddings, p=2, dim=1)
        topic_mu = F.normalize(topic_mu, p=2, dim=1)
        cosine_sim = torch.matmul(word_embeddings, topic_mu.T)
        C = 1.0 - cosine_sim

        log_pi = -C / 0.1
        log_pi = log_pi - log_pi.logsumexp(dim=1, keepdim=True)
        pi_star = log_pi.exp()

        for _ in range(sinkhorn_iter):
            pi_star = pi_star / (pi_star.sum(dim=1, keepdim=True) + epsilon)
            pi_star = pi_star / (pi_star.sum(dim=0, keepdim=True) + epsilon)

        kappa_inv = 1.0 / (topic_kappa + epsilon)
        weighted_loss = C * pi_star * kappa_inv.view(1, -1)

        loss = weighted_loss.sum() / word_embeddings.size(0)

        return loss

    def get_topics(self):
        return self.beta

    def forward(self, z):
        self.theta = z
        self.beta = self.beta_softmax(self.log_vmf_pdf(self.mu_mat, self.kappa, self.word_embeddings_mat))
        logits = torch.log(torch.matmul(self.theta, self.beta) + 1e-10)
        vEC_loss = self.vEC_loss(self.word_embeddings_mat, self.mu_mat, self.kappa)

        return logits, vEC_loss

class vMNVTM:
    def __init__(self, gamma = 1e-10, kappa1 = 10, kappa2 = 10, epochs=50, batch_size=256, gpu_num=1, numb_embeddings=20,
                 learning_rate=0.002, weight_decay=1.2e-6, penalty=1, temp=10, lambda_sh=0.1,
                 top_n_words=25, num_representative_docs=5, top_n_topics=100, embedding_dim=100):

        self.dataset = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu_num = gpu_num
        self.numb_embeddings = numb_embeddings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.penalty = penalty
        self.top_n_words = top_n_words
        self.num_representative_docs = num_representative_docs
        self.top_n_topics = top_n_topics
        self.embedding_dim = embedding_dim
        self.bow = None
        self.device = torch.device('cuda')
        self.temp = temp
        self.z = None
        self.model = None
        self.lambda_sh = lambda_sh
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.gamma = gamma

    def train_model(self, dataset, testset):

        if self.z is None:
            self.fit_transform(dataset, testset)

        # create model output
        model_output = {}
        model_output['topics'] = [i[:self.top_n_words] for i in self.topics]
        model_output['topic-word-matrix'] = self.model.decoder.get_topics().cpu().detach().numpy()
        model_output['topic-document-matrix'] = self.z.T
        model_output['test-topic-document-matrix'] = self.z2.T

        return model_output

    def fit_transform(self, dataset, testset):

        combined_dataset = dataset + testset
        self.dataset = dataset

        print("\ntransforming bag_of_words..")
        self.tp = TextProcessor(combined_dataset)
        self.tp.process()
        combined_bow = torch.tensor(self.tp.bow)

        self.bow = combined_bow[:len(dataset)]
        self.test_bow = combined_bow[len(dataset):]

        bag_of_words = self.bow

        print("end transforming bag_of_words")
        print("bow of train:", self.bow.shape)
        print("bow of test:", self.test_bow.shape)

        layer = bag_of_words.shape[1] // 16
        hidden = get_mlp([bag_of_words.shape[1], bag_of_words.shape[1] // 4, layer], nn.GELU)
        h_to_z = nn.Softmax()

        embedding = nn.Embedding(bag_of_words.shape[1], 100)
        print("loading glove..")
        glove_vectors = KeyedVectors.load('./data/glove-wiki-gigaword-100.model')
        print("end loading glove")
        embed = np.asarray([glove_vectors[self.tp.index_to_word[i]]
                            if self.tp.index_to_word[i] in glove_vectors
                            else np.asarray([1] * 100) for i in self.tp.index_to_word])
        print("embed.shape:", embed.shape)
        embedding.weight = torch.nn.Parameter(torch.from_numpy(embed).float())
        embedding.weight.requires_grad = True

        topics = Generative_model(word_embedding = embedding,
                                  n_topics = self.numb_embeddings,
                                  topic_embeddings_size = 100,
                                  kappa = self.kappa2).to(self.device)

        self.model = VNTM(hidden=hidden,
                          h_to_z=h_to_z,
                          topics=topics,
                          layer=layer,
                          top_number=self.numb_embeddings,
                          penalty=self.penalty,
                          temp=self.temp,
                          lambda_sh=self.lambda_sh,
                          kappa = self.kappa1,
                          gamma = self.gamma
                          ).to(self.device).float()

        total_params = sum(p.numel() for p in self.model.parameters())

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.002,
                                                             steps_per_epoch=int(
                                                                 bag_of_words.shape[0] / self.batch_size) + 1,
                                                             epochs=self.epochs)

        print("start training..")
        for epoch in range(self.epochs):
            print("epoch", epoch + 1, ':')
            self.train(bag_of_words, self.batch_size)

        # save topics
        emb = self.model.decoder.get_topics().cpu().detach().numpy()
        self.topics = [[self.tp.index_to_word[ind] for ind in np.argsort(emb[i])[::-1][:self.top_n_topics]] for i in
                       range(self.numb_embeddings)]
        self.topics_score = [[score for score in np.sort(emb[i])[::-1]] for i in range(self.numb_embeddings)]

        data_batch = bag_of_words.float()
        self.model.cpu()

        z = self.model.hidden(data_batch)
        z_mean = self.model.fc_mean(z)
        self.ptrain_emb = z_mean
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        self.z = self.model.h_to_z(z_mean).detach().numpy()
        self.ptrain_label = np.argmax(self.z, axis=1)

        self.topic_doc = [[ind for ind in np.argsort(self.z[:, i])[::-1][:100]] for i in
                          range(self.numb_embeddings)]
        self.topic_doc_score = [[ind for ind in np.sort(self.z[:, i])[::-1][:100]] for i in
                                range(self.numb_embeddings)]

        z2 = self.model.hidden(self.test_bow.float())
        z2_mean = self.model.fc_mean(z2)
        self.ptest_emb = z2_mean
        z2_mean = z2_mean / z2_mean.norm(dim=-1, keepdim=True)
        self.z2 = self.model.h_to_z(z2_mean).detach().numpy()
        self.ptest_label = np.argmax(self.z2, axis=1)

        return self.topics, self.z

    def train(self, bow, batch_size):

        self.model.train()
        total_loss = 0.0

        indices = torch.randperm(bow.shape[0])
        indices = torch.split(indices, batch_size)
        length = len(indices)

        for idx, ind in enumerate(indices):
            data_batch = bow[ind].to(self.device).float()

            d = self.model(x=data_batch, device=self.device)

            total_loss += d['loss'].sum().item() / batch_size
            loss = d['loss']

            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()
            self.scheduler.step()

        print(total_loss / length)

    def get_topics(self, index):
        return [(i, j) for i, j in zip(self.topics[index], self.topics_score[index])][:self.top_n_words]

    def topic_word_matrix(self):
        return self.model.decoder.get_topics().cpu().detach().numpy()

    def topic_keywords(self):
        return self.topics

    def get_document_info(self, top_n_words=10):
        data = []
        for topic_id in range(self.numb_embeddings):
            topic_keywords = self.get_topics(topic_id)[:top_n_words]
            topic_keywords_str = "_".join([word for word, _ in topic_keywords[:3]])

            doc_indices = np.argsort(self.z[:, topic_id])[::-1]
            representative_doc_index = doc_indices[0]
            representative_doc = self.dataset[representative_doc_index]

            dominant_topics = np.argmax(self.z, axis=1)
            num_docs = np.sum(dominant_topics == topic_id)

            data.append(
                [topic_id, f"{topic_id}_{topic_keywords_str}", topic_keywords_str, representative_doc, num_docs])

        df = pd.DataFrame(data, columns=["Topic", "Name", "Top_n_words", "Representative_Doc", "Num_Docs"])
        return df

    def calculate_perplexity(self):
        self.model.to(self.device)
        self.model.eval()
        bag_of_words = self.test_bow
        with torch.no_grad():
            total_log_likelihood = 0
            total_word_count = 0

            for doc in bag_of_words:
                doc = doc.to(self.device).float()
                doc_sum = doc.sum().item()
                if doc_sum == 0:
                    continue

                log_prob = self.model.forward(doc.unsqueeze(0),self.device)['log_prob']
                log_likelihood = (doc * log_prob).sum().item()
                total_log_likelihood += log_likelihood
                total_word_count += doc_sum

            perplexity = np.exp(-total_log_likelihood / total_word_count)
            return perplexity
