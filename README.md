# Ponderada Translation With a Transformer

Repositório com a ponderada de programação "Atividade: Tradução usando Transformer com Controle de Versões" do módulo 11 de ciência da computação.

## Descrição do Projeto

Implementação de um modelo Transformer seguindo o tutorial oficial do TensorFlow para tradução automática de português para inglês. O modelo utiliza a arquitetura encoder-decoder com mecanismo de atenção multi-head, baseado no paper "Attention Is All You Need".

## Arquitetura Implementada

### Componentes Principais

- **Encoder**: 4 camadas com atenção multi-head e redes feed-forward
- **Decoder**: 4 camadas com atenção mascarada e cross-attention
- **Embedding**: Dimensão 128 com codificação posicional
- **Tokenização**: Subpalavras usando BertTokenizer otimizado

### Hiperparâmetros

```python
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
batch_size = 64
epochs = 20
```

## Dataset e Treinamento

**Dataset**: TED Talks PT-EN (~50.000 exemplos de treinamento)

- Treinamento: ~50k pares de sentenças
- Validação: ~1.1k exemplos
- Teste: ~2k exemplos

**Resultados de Treinamento**:

- **Época 1**: Loss 6.70, Acurácia 11.39%
- **Época 10**: Loss 2.11, Acurácia 58.25%
- **Época 20**: Loss 1.45, Acurácia 67.99%

## Exemplos de Tradução

| Português | Predição | Ground Truth |
|-----------|----------|--------------|
| "este é um problema que temos que resolver." | "this is a problem we have to solve ." | "this is a problem we have to solve ." |
| "os meus vizinhos ouviram sobre esta ideia." | "my neighbors heard about this idea ." | "and my neighboring homes heard about this idea ." |
