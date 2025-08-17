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

## Análise: Pontos Positivos e Negativos

### ✅ **Pontos Positivos**

#### **Arquitetura e Desempenho**

- **Paralelização Superior**: Diferente de RNNs, processa tokens simultaneamente, resultando em treinamento mais rápido
- **Dependências de Longo Alcance**: Captura relações entre palavras distantes sem degradação de gradiente
- **Flexibilidade**: Não assume relações temporais/espaciais específicas nos dados
- **Generalização**: Modelo consegue transliterar palavras desconhecidas (ex: "triceratops" → "trigatotys")

#### **Implementação**

- **Código Limpo**: Implementação modular e bem estruturada
- **Visualizações**: Mapas de atenção permitem interpretabilidade do modelo
- **Exportação**: Modelo pode ser salvo como SavedModel para produção
- **Didático**: Código comentado facilita compreensão da arquitetura

### ❌ **Pontos Negativos**

#### **Limitações Computacionais**

- **Consumo de Memória**: Atenção quadrática O(n²) consome muita RAM para sequências longas
- **Hiperparâmetros Reduzidos**: Para viabilizar execução, modelo usa apenas 128 dimensões (vs 512 do paper original)
- **Dataset Limitado**: ~50k exemplos é pequeno para modelos Transformer de qualidade produção

#### **Limitações de Design**

- **Sem Transfer Learning**: Não aproveita modelos pré-treinados (BERT, T5, etc.)
- **Sem Beam Search**: Usa apenas greedy decoding, limitando qualidade das traduções
- **Codificação Posicional Fixa**: Sinusoidal pode ser limitante para sequências muito longas

#### **Limitações Práticas**

- **Overfitting**: Sem regularização avançada além de dropout
- **Avaliação Simplificada**: Não usa métricas padrão como BLEU score
- **Inferência Sequencial**: Decoder ainda é sequencial, limitando velocidade de inferência

## Comparação CPU vs GPU

### **Treinamento em CPU**

- **Tempo por Época**: ~45-58 segundos (baseado nos logs)
- **Limitações**:
  - Processamento sequencial de operações matriciais
  - Gargalo em multiplicações de matriz grandes
  - Memória limitada para batches maiores
- **Viabilidade**: Apenas para prototipagem e datasets pequenos

### **Treinamento em GPU**

- **Vantagens Esperadas**:
  - **Speedup 10-50x**: Operações matriciais altamente paralelizáveis
  - **Batches Maiores**: Mais memória permite batch_size > 64
  - **Hiperparâmetros Realistas**: Possibilita d_model=512, num_layers=6
- **Requisitos**: GPU com ≥8GB VRAM para modelo completo

### **Recomendações**

- **CPU**: Adequada apenas para aprendizado e debugging
- **GPU**: Essencial para treinamento sério e modelos maiores
- **Cloud**: TPUs (Colab Pro) ideais para experimentação rápida

## Melhorias Propostas

1. **Transfer Learning**: Usar modelos pré-treinados como mT5
2. **Beam Search**: Implementar busca em feixe para melhor qualidade
3. **Métricas Padrão**: Adicionar BLEU, ROUGE, BERTScore
4. **Regularização**: Label smoothing, weight decay
5. **Arquitetura**: Testar Transformer-XL ou GPT-style unidirecionais

## Estrutura do Repositório

```
├── README.md
├── transformer_training.ipynb    # Notebook principal
├── checkpoints/                  # Checkpoints do modelo
├── translator/                   # Modelo exportado
└── requirements.txt             # Dependências
```

## Executar o Projeto

```bash
pip install tensorflow tensorflow-datasets tensorflow-text
python transformer_training.py
```

## Conclusão

O tutorial demonstra efetivamente os conceitos fundamentais da arquitetura Transformer, proporcionando uma base sólida para compreensão de modelos modernos de NLP. Embora as limitações computacionais impeçam resultados de produção, a implementação serve como excelente ferramenta educacional e ponto de partida para projetos mais ambiciosos.
