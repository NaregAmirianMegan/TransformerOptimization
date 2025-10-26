# Optimizing a Transformer

Walkthrough of implementing a transformer in pytorch, introducing newer transformer
features, and optimizing it down to the kernel level.

* Fundamental Transformer Architecture:

Series of Multi-headed attention blocks and MLP blocks


* Multi-headed attention:

Given a single input sequence E, each attention head computes a delta_E_i on E each
E_i in that sequence

* Single-attention head:

* Defined by 3 weight matrices W_q, W_k, W_v
* W_q generates queries: W_q * E
* W_k generates keys: W_k * E
* W_v generates values: W_v * E

E : (n, t) (each E_i is n-dim, t E_i per seq.)

W_q : (l_dim, n)
W_k : (l_dim, n)
W_v : (n, n) -> linear decomp. into W_v_up : (n, ) * W_v_down : (v_dim, n)

Q = W_q * E
K = W_k * E
V = W_v_up * W_v_down * E

Q : (l_dim, t), where each column is query for corresponding input col. vec. E_i
K : (l_dim, t), where each column is key for corresponding input col. vec. E_i
V : (n, t), where each column is value for corresponding input col. vec. E_i

Now we need to perform a dot product between the columns of Q and columns of K to 
generate attention matrix.

A = K^T * V

A : (t, t)

Now we apply appropriate masking, and a scaling factor to A and softmax to the columns

A_norm = colwise_softmax(mask(A)/factor)

The last step is to compute the updates delta_E for each E_i. This is done by summing the
value vectors for each column of A_norm scaling by the values in each row in that column.

E_delta = V * A_norm : (n, t)


* Back to Multi-head attention

An E_delta is computed for each attention head and accumulated back into E

for each head i:
	E += E_delta_i


* Multi-Layer Perceptron

Each attention head returns a sequence E composed of E_i n-dim vectors

Each vector E_i gets passed through an MLP composed of a linear layer, non-linearity, 
and another linear layer to produce an output E_i_prime. This E_prime seq. then gets fed
into another attention layer...

This block would look like the following

First layer:

W_up : (h_dim, n)
B : (h_dim, 1), duplicatively added to each column in H

H = W_up * E + B: (h_dim, t)
H = NonLin(H)

W_down : (m, h_dim)
B : (m, 1), duplicatively added to each column in H
E_prime = W_down * H + B

E_prime : (m, t), new seq. of t, m-dim vectors to be fed into attention block



Implementation Plan: 

- Implement very basic transformer architecture
- Perform basic optimizations from original paper
- Perform further more arch modern optimizations
- Perform kernel optimizations and achieve better performance
"""