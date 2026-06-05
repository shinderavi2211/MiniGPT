import torch
import torch.nn as nn
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset Loaded:")
print(text)

chars = sorted(list(set(text)))

print("\nVocabulary:")
print(chars)

print("\nVocabulary Size:", len(chars))
# Character → Number

stoi = {ch:i for i,ch in enumerate(chars)}

# Number → Character

itos = {i:ch for i,ch in enumerate(chars)}

# Encode function

def encode(text):
    return [stoi[c] for c in text]

# Decode function

def decode(tokens):
    return ''.join([itos[i] for i in tokens])

sample = "hello"

encoded = encode(sample)

print("\nEncoded:")
print(encoded)

decoded = decode(encoded)

print("\nDecoded:")
print(decoded)
# Convert tokens to tensor

tokens = torch.tensor(encoded)

print("\nTensor:")
print(tokens)

# Embedding Layer

embedding = nn.Embedding(
    num_embeddings=len(chars),
    embedding_dim=8
)

embedded = embedding(tokens)

print("\nEmbedding Shape:")
print(embedded.shape)

print("\nEmbeddings:")
print(embedded)
# Add batch dimension

x = embedded.unsqueeze(0)

print("\nInput Shape:")
print(x.shape)

# Self Attention

attention = nn.MultiheadAttention(
    embed_dim=8,
    num_heads=1,
    batch_first=True
)

attn_output, attn_weights = attention(
    x,
    x,
    x
)

print("\nAttention Output Shape:")
print(attn_output.shape)

print("\nAttention Weights Shape:")
print(attn_weights.shape)
# Transformer Block

transformer = nn.TransformerEncoderLayer(
    d_model=8,
    nhead=1,
    batch_first=True
)

transformer_output = transformer(x)

print("\nTransformer Output Shape:")
print(transformer_output.shape)
# Prediction Head

lm_head = nn.Linear(
    8,
    len(chars)
)

logits = lm_head(
    transformer_output
)

print("\nLogits Shape:")
print(logits.shape)
# Training Data

input_text = "hell"
target_text = "ello"

x_train = torch.tensor(
    encode(input_text)
).unsqueeze(0)

y_train = torch.tensor(
    encode(target_text)
)

print("\nInput Tokens:")
print(x_train)

print("\nTarget Tokens:")
print(y_train)
# Forward Pass

embedded = embedding(x_train)

transformer_output = transformer(
    embedded
)

logits = lm_head(
    transformer_output
)

print("\nTraining Logits Shape:")
print(logits.shape)
# Loss Function

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.view(-1, len(chars)),
    y_train
)

print("\nLoss:")
print(loss.item())
# Optimizer

optimizer = torch.optim.Adam(
    list(embedding.parameters()) +
    list(transformer.parameters()) +
    list(lm_head.parameters()),
    lr=0.01
)

# One Training Step

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\nOne Training Step Completed!")
print("\nStarting Training...\n")

for epoch in range(500):

    embedded = embedding(x_train)

    transformer_output = transformer(
        embedded
    )

    logits = lm_head(
        transformer_output
    )

    loss = loss_fn(
        logits.view(-1, len(chars)),
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch} | Loss = {loss.item():.4f}"
        )
print("\n=== TEXT GENERATION ===")

# Input
test_text = "hell"

x_test = torch.tensor(
    encode(test_text)
).unsqueeze(0)

with torch.no_grad():

    embedded = embedding(x_test)

    transformer_output = transformer(
        embedded
    )

    logits = lm_head(
        transformer_output
    )

    next_token = torch.argmax(
        logits[0, -1]
    ).item()

predicted_char = decode(
    [next_token]
)

print("Input :", test_text)
print("Predicted Next Character :", predicted_char)

print("Generated Text :",
      test_text + predicted_char)
