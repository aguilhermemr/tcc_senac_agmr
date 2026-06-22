# Criação do repositório GitHub `tcc_senac_agmr`

Este arquivo descreve os passos para publicar o conteúdo desta pasta como o
repositório `tcc_senac_agmr` no GitHub.

## Pré-requisitos

- Conta no GitHub (https://github.com)
- Git instalado localmente (`git --version` para conferir)
- Esta pasta `angelia/` baixada para sua máquina

## Passo 1 — Criar o repositório vazio no GitHub

1. Acesse https://github.com/new
2. Em **Repository name**, digite: `tcc_senac_agmr`
3. **Description** (sugestão):

       PPO+GAT pipeline para otimização de workflows BPMN —
       código de apoio ao TCC SENAC 2026

4. Marque **Public** (recomendado, para a banca acessar)
5. **NÃO** marque "Initialize this repository with README" — vamos subir o conteúdo já pronto
6. Clique em **Create repository**

## Passo 2 — Subir o conteúdo local

Abra um terminal nesta pasta `angelia/` e execute:

    cd /caminho/para/angelia
    git init
    git add .
    git commit -m "Initial commit: PPO+GAT pipeline (versão piloto - TCC Senac 2026)"
    git branch -M main
    git remote add origin https://github.com/SEU-USUARIO/tcc_senac_agmr.git
    git push -u origin main

Substitua `SEU-USUARIO` pelo seu nome de usuário real do GitHub (no documento
está como `agmr`; ajuste se for diferente).

## Passo 3 — Capturar o identificador do commit

Após o push, anote o hash completo do commit (será o identificador a citar na defesa):

    git log --format="%H" -n 1

Inclua esse hash na defesa, como prometido na seção 5.4.2 do TCC.

## Passo 4 — Atualizar a URL no TCC

Se o seu usuário GitHub for diferente de `agmr`, abra o arquivo .docx e
substitua todas as ocorrências de `agmr/tcc_senac_agmr` pela URL correta
(use Ctrl+H no Word).

> Estado atual do TCC entregue ao coordenador (rev 7): as 4 URLs do GitHub
> foram **removidas** e generalizadas para "repositório do projeto, a ser
> informado no momento da defesa", justamente para permitir flexibilidade
> sobre o nome de usuário final. Se quiser reincluir as URLs no documento,
> me avise.

## Passo 5 — Tag de versão (opcional, recomendado)

Para marcar a versão entregue ao coordenador:

    git tag -a v0.1.0-pilot -m "Versao piloto entregue ao coordenador (500k timesteps, RTX 3060)"
    git push origin v0.1.0-pilot

E quando rodar a escala plena:

    git tag -a v1.0.0-final -m "Versao final (2M timesteps, resultados consolidados)"
    git push origin v1.0.0-final

## Estrutura do repo após push

    tcc_senac_agmr/
    ├── README.md              # como rodar
    ├── LICENSE                # MIT
    ├── CITATION.cff           # como citar o trabalho
    ├── .gitignore             # arquivos ignorados (mlruns/, .venv/, dados gerados)
    ├── requirements.txt       # dependências pinned
    ├── configuracoes/
    │   └── default.yaml       # hiperparâmetros PPO+GAT+recompensa
    ├── exemplos/
    │   ├── process_A.bpmn     # Selecao de Pessoal (linear)
    │   ├── process_B.bpmn     # Compras Corporativas (paralelismo)
    │   └── process_C.bpmn     # Gestao de Incidentes (loops de escalonamento)
    ├── documentacao/
    │   ├── CAPITULO_5_TEMPLATE.md
    │   └── ROADMAP.md
    └── src/
        ├── bpmn_loader.py     # BPMN XML -> grafo (PyTorch Geometric)
        ├── case_generator.py  # geracao 70/20/10
        ├── reward.py          # funcao R(s,a) multi-objetivo
        ├── workflow_env.py    # ambiente Gymnasium token-passing
        ├── gat_policy.py      # GATv2 3 camadas + 4 heads
        ├── baselines.py       # FIFO + prioridade-fixa
        └── train.py           # entry point com MLflow

## Verificação rápida pós-push

Acesse `https://github.com/SEU-USUARIO/tcc_senac_agmr` e confira:

- [ ] README.md aparece formatado corretamente na página inicial
- [ ] A licença MIT é detectada automaticamente (badge no topo direito)
- [ ] CITATION.cff produz um botão "Cite this repository" no menu lateral
- [ ] Os três arquivos .bpmn estão em `exemplos/`
- [ ] `requirements.txt` está visível

## Em caso de dúvida sobre autenticação

Se o `git push` pedir senha e rejeitar, é porque o GitHub não aceita mais
senha desde 2021. Use um Personal Access Token:

1. https://github.com/settings/tokens → Generate new token (classic)
2. Marque o escopo `repo`
3. Use o token como senha quando o git pedir

Ou configure o GitHub CLI (`gh auth login`) para uma autenticação mais simples.
