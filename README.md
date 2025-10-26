# QML-GA Project — Variational Quantum Classifiers otimizados por Algoritmos Genéticos

Repositório para experimentos com **Variational Quantum Classifiers (VQC)** usando **Algoritmos Genéticos (AG)** e otimizadores por gradiente (Adam/Nesterov) com **PennyLane**.  
A pipeline varre automaticamente **datasets**, **feature maps** (Amplitude/ZZ), **ansätze** (`ansatz_1 … ansatz_6`) e **hiperparâmetros**, registrando **resultados reprodutíveis**.

---

## Sumário
- [Estrutura do projeto](#estrutura-do-projeto)
- [Ambiente de desenvolvimento](#ambiente-de-desenvolvimento)
  - [Pré-requisitos](#pré-requisitos)
  - [Criar o ambiente com Poetry](#criar-o-ambiente-com-poetry)
  - [VS Code](#vs-code)
- [Configurações (YAML)](#configurações-yaml)
  - [Datasets](#datasets)
  - [Feature Maps](#feature-maps)
  - [Ansätze](#ansätze)
  - [Hipergrids](#hipergrids)
  - [Devices e Optimizers (opcionais)](#devices-e-optimizers-opcionais)
- [Dados & pré-processamento](#dados--pré-processamento)
- [Como rodar os experimentos](#como-rodar-os-experimentos)
  - [Dry-run (validação/listagem)](#1-dry-run-validaçãolistagem)
  - [Treinando de fato](#2-treinando-de-fato)
  - [Flags úteis](#flags-úteis)
- [Saídas e avaliação](#saídas-e-avaliação)

---

## Estrutura do projeto

