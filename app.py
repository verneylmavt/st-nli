import os
import json
import streamlit as st
import torch
from torch import nn
from torch.nn import functional as F

# ----------------------
# Model Information
# ----------------------
model_info = {
    "decomposable-attention": {
        "subheader": "Model: Decomposable Attention Model",
        "pre_processing": """
Dataset = Stanford Natural Language Inference (SNLI)
Embedding Model = GloVe("6B.100d")
        """,
        "parameters": """
Batch Size = 256
Embedding Size = 100
Feedforward Hidden Size = 200
Learning Rate = 0.0005
Epochs = 15
Optimizer = Adam
Loss Function = CrossEntropyLoss
        """,
        "model_code": """
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
    
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)
    def forward(self, A, B):
        f_A = self.f(A)
        f_B = self.f(B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
    
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)
    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B

class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)
    def forward(self, V_A, V_B):
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat

class Model(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)
    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
        """,
        "forward_pass": {
        "Embedding (Premises and Hypotheses)": r'''
        \mathbf{A} = \text{Embedding}(\text{premises}) \quad \mathbf{B} = \text{Embedding}(\text{hypotheses}) \\~~\\
        \mathbf{A}, \mathbf{B} \in \mathbb{R}^{b \times n \times d}
        ''',
        "Attend Phase": r'''
        \mathbf{f}_A = f(\mathbf{A}) \quad \mathbf{f}_B = f(\mathbf{B}) \\~~\\
        \mathbf{f}_A \in \mathbb{R}^{b \times n \times h}, \quad \mathbf{f}_B \in \mathbb{R}^{b \times n \times h} \\~~\\ \underline{\hspace{7.5cm}} \\~~\\
        \mathbf{e} = \mathbf{f}_A \cdot \mathbf{f}_B^\top \\~~\\
        \mathbf{e} \in \mathbb{R}^{b \times n \times n} \\~~\\ \underline{\hspace{7.5cm}} \\~~\\
        \boldsymbol{\beta} = \text{softmax}(\mathbf{e}) \cdot \mathbf{B} \quad \boldsymbol{\alpha} = \text{softmax}(\mathbf{e}^\top) \cdot \mathbf{A} \\~~\\
        \boldsymbol{\beta}, \boldsymbol{\alpha} \in \mathbb{R}^{b \times n \times d}
        ''',
        "Compare Phase": r'''
        \mathbf{V}_A = g([\mathbf{A}, \boldsymbol{\beta}]) \quad \mathbf{V}_B = g([\mathbf{B}, \boldsymbol{\alpha}]) \\~~\\
        \mathbf{V}_A, \mathbf{V}_B \in \mathbb{R}^{b \times n \times h}
        ''',
        "Aggregate Phase": r'''
        \bar{\mathbf{V}}_A = \sum_{i=1}^{n} \mathbf{V}_A[:, i, :] \quad \bar{\mathbf{V}}_B = \sum_{j=1}^{n} \mathbf{V}_B[:, j, :] \\~~\\
        \bar{\mathbf{V}}_A, \bar{\mathbf{V}}_B \in \mathbb{R}^{b \times h} \\~~\\ \underline{\hspace{7.5cm}} \\~~\\
        \mathbf{H} = h([\bar{\mathbf{V}}_A, \bar{\mathbf{V}}_B]) \\~~\\
        \mathbf{H} \in \mathbb{R}^{b \times h} \\~~\\ \underline{\hspace{7.5cm}} \\~~\\
        \mathbf{Y}_{\text{hat}} = \text{Linear}(\mathbf{H}) \\~~\\
        \mathbf{Y}_{\text{hat}} \in \mathbb{R}^{b \times c}
        '''
        }
    }
}

# @st.cache_resource
def load_model(model_name, vocab):
    def mlp(num_inputs, num_hiddens, flatten):
        net = []
        net.append(nn.Dropout(0.2))
        net.append(nn.Linear(num_inputs, num_hiddens))
        net.append(nn.ReLU())
        if flatten:
            net.append(nn.Flatten(start_dim=1))
        net.append(nn.Dropout(0.2))
        net.append(nn.Linear(num_hiddens, num_hiddens))
        net.append(nn.ReLU())
        if flatten:
            net.append(nn.Flatten(start_dim=1))
        return nn.Sequential(*net)
    
    class Attend(nn.Module):
        def __init__(self, num_inputs, num_hiddens, **kwargs):
            super(Attend, self).__init__(**kwargs)
            self.f = mlp(num_inputs, num_hiddens, flatten=False)
        def forward(self, A, B):
            f_A = self.f(A)
            f_B = self.f(B)
            e = torch.bmm(f_A, f_B.permute(0, 2, 1))
            beta = torch.bmm(F.softmax(e, dim=-1), B)
            alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
            return beta, alpha
        
    class Compare(nn.Module):
        def __init__(self, num_inputs, num_hiddens, **kwargs):
            super(Compare, self).__init__(**kwargs)
            self.g = mlp(num_inputs, num_hiddens, flatten=False)
        def forward(self, A, B, beta, alpha):
            V_A = self.g(torch.cat([A, beta], dim=2))
            V_B = self.g(torch.cat([B, alpha], dim=2))
            return V_A, V_B

    class Aggregate(nn.Module):
        def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
            super(Aggregate, self).__init__(**kwargs)
            self.h = mlp(num_inputs, num_hiddens, flatten=True)
            self.linear = nn.Linear(num_hiddens, num_outputs)
        def forward(self, V_A, V_B):
            V_A = V_A.sum(dim=1)
            V_B = V_B.sum(dim=1)
            Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
            return Y_hat
    class DecomposableAttention(nn.Module):
        def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                    num_inputs_compare=200, num_inputs_agg=400, **kwargs):
            super(DecomposableAttention, self).__init__(**kwargs)
            self.embedding = nn.Embedding(len(vocab), embed_size)
            self.attend = Attend(num_inputs_attend, num_hiddens)
            self.compare = Compare(num_inputs_compare, num_hiddens)
            self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)
        def forward(self, X):
            premises, hypotheses = X
            A = self.embedding(premises)
            B = self.embedding(hypotheses)
            beta, alpha = self.attend(A, B)
            V_A, V_B = self.compare(A, B, beta, alpha)
            Y_hat = self.aggregate(V_A, V_B)
            return Y_hat
    try:
        net = DecomposableAttention(vocab, 100, 200)
        net.load_state_dict(torch.load(os.path.join("models", "model-state.pt"), weights_only=True, map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model for {model_name}: {e}")
        st.stop()
    return net

# @st.cache_resource
def load_vocab(model_name):
    try:
        with open(os.path.join("models", "vocab-dict.json"), 'r') as json_file:
            vocab = json.load(json_file)
        return vocab
    except FileNotFoundError:
        st.error(f"Vocabulary file not found for {model_name}.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the vocabulary for {model_name}: {e}")
        st.stop()

# ----------------------
# Prediction Function
# ----------------------
def predict_snli(net, vocab, premises, hypotheses):
    premise = premises.strip().split()
    hypothesis = hypotheses.strip().split()
    premise_indices = [vocab.get(word, vocab["<pad>"]) for word in premise]
    hypothesis_indices = [vocab.get(word, vocab["<pad>"]) for word in hypothesis]
    net.eval()
    premise_tensor = torch.tensor(premise_indices).unsqueeze(0)
    hypothesis_tensor = torch.tensor(hypothesis_indices).unsqueeze(0)
    with torch.no_grad():
        outputs = net((premise_tensor, hypothesis_tensor))
        label_idx = torch.argmax(outputs, dim=1).item()
    return "entailment" if label_idx == 0 else "contradiction" if label_idx == 1 else "neutral"

# ----------------------
# Page UI
# ----------------------
def main():
    st.title("Natural Language Inference")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    
    vocab = load_vocab(model)
    net = load_model(model, vocab)
    
    st.subheader(model_info[model]["subheader"])
    user_input_1 = st.text_area("Enter Text Here (Premise):")
    user_input_2 = st.text_area("Enter Text Here (Hypothesis):")
    
    if st.button("Analyze"):
        if user_input_1 and user_input_2:
            with st.spinner('Analyzing...'):
                inference = predict_snli(net, vocab, user_input_1, user_input_2)
            if inference == 'entailment':
                st.success(f"**Inference:** {inference.capitalize()}")
            elif inference == 'neutral':
                st.warning(f"**Inference:** {inference.capitalize()}")
            else:
                st.error(f"**Inference:** {inference.capitalize()}")
        else:
            st.warning("Please enter some text for inference.")
            
    st.feedback("thumbs")
    st.warning("""Check here for more details: [GitHub Repo🐙](https://github.com/verneylmavt/st-nli)""")
    st.divider()
    
    st.subheader("""Pre-Processing""")
    st.code(model_info[model]["pre_processing"], language="None")
    
    st.subheader("""Parameters""")
    st.code(model_info[model]["parameters"], language="None")
    
    st.subheader("""Model""")
    st.code(model_info[model]["model_code"], language="python")
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key)
            st.latex(value)
    else: pass

if __name__ == "__main__":
    main()