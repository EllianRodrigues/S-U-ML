## S(U)LM: Evaluation of Semantic Preservation in Multi-Agent Pipelines with Small Language Models for UML

Projeto para avaliar preservação semântica em pipelines de modelos pequenos aplicados à geração e interpretação de UML.

O pipeline principal é: texto -> use case -> UML -> texto -> comparação com o texto inicial. A ideia é testar combinações de modelos para extrair o melhor fluxo (qual gera use case e UML que preservam mais o significado, a um menor custo).

## **Estrutura do Repositório**

- **Models:** implementações e wrappers por família de modelos. Exemplos: [models/uml_generation/Qwen2_5_3B_Instruct.py](models/uml_generation/Qwen2_5_3B_Instruct.py), [models/requirement_extraction/Qwen2_5_5B.py](models/requirement_extraction/Qwen2_5_5B.py)
- **Agents:** agentes que orquestram passos do pipeline (invocam modelos e fazem pós-processamento). Exemplo: [agents/UML/UML.py](agents/UML/UML.py)
- **Prompts:** templates usados para cada passo; edite-os aqui para mudar comportamento das gerações. Exemplo: [prompts/UML.json](prompts/UML.json)
- **Config:** arquivo com o texto de entrada inicial e opções básicas do pipeline: [config.toml](config.toml)

Obs: no futuro haverá mais modelos adicionados nas pastas `models/*` — a arquitetura foi pensada para facilitar a integração de novas variantes.

## **O que cada pasta contém (resumido)**

- **models/**: cada subpasta agrupa wrappers e utilitários para um modelo específico e a lógica de tokenização/geração.
- **agents/**: alto nível — funções reutilizáveis que chamam os modelos com prompts e fazem limpeza/validação de saída.
- **prompts/**: JSONs com instruções; trocar o prompt aqui altera o comportamento sem tocar no código.
- **main.py**: orquestrador. Executa o pipeline completo usando os agentes e modelos configurados.

## **Fluxo do Pipeline**

- Entrada: texto (use case / descrição inicial) — definido em [config.toml](config.toml).
- Etapa 1: extração de `Use Case` (models/requirement_extraction)
- Etapa 2: geração de UML a partir do Use Case (models/uml_generation)
- Etapa 3: interpretação do UML de volta em texto (models/uml_interpretation)
- Etapa 4: comparação semântica entre texto inicial e texto reconstruído (models/sentence_transformers)

O objetivo é comparar qual combinação de modelos preserva melhor o significado (maior similaridade textual) e qual combinação é mais barata (menos tokens/tempo).

## **Métricas e extensões futuras**

- Atualmente a comparação é feita por similaridade de sentenças usando `sentence-transformers` (`all-MiniLM-L6-v2`): o texto inicial e o texto reconstruído são convertidos em embeddings e comparados por similaridade de cosseno (score entre 0 e 1; valores mais altos = mais similares). Opcionalmente pode-se usar distância euclidiana ou thresholds para decisões binárias.
- Em breve será possível coletar e exportar métricas adicionais como: token usage, tempo de inferência, e scores agregados.
- O `config.toml` será estendido para permitir seleção de modelo por etapa (ex.: escolher modelo de extração vs modelo de geração de UML) e outras opções de execução.

## **Como Rodar**

1. Crie e ative o ambiente (recomendado Conda: ):

```bash
conda create -n amaspput python=3.10 -y
conda activate amaspput
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 (recomendado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Execução simples:

```powershell
python .\main.py
```

Isso irá executar o pipeline usando os agentes e modelos atualmente configurados.

## **Dicas para experimentação**

- Para alterar o comportamento de geração edite os arquivos em `prompts/` (por exemplo [prompts/UML.json](prompts/UML.json)).
- Para adicionar novos modelos, crie um novo wrapper em `models/` e conecte-o através de um `agent` ou atualize `main.py`.
- Para comparar variantes, você pode alterar `config.toml` (quando suportado) para escolher modelos diferentes por etapa.

## **Propósito e objetivos de avaliação**

- A meta é encontrar a melhor combinação de modelos para gerar `use case` e `UML`, medindo se o texto final é semanticamente próximo ao texto inicial e qual combinação usa menos recursos.



