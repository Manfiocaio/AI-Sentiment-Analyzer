# 🤖 AI Sentiment Analyzer

Aplicação web de Inteligência Artificial que analisa o sentimento de frases em tempo real, utilizando Machine Learning com Python e Flask.

O sistema classifica textos como **Positivo** ou **Negativo**, exibindo também a confiança da IA e um histórico de análises, com interface moderna, responsiva e focada em UX profissional.

---

## 🚀 Demonstração do Projeto

Digite uma frase como:

* "Hoje foi um dia incrível"
* "Estou muito cansado"
* "Isso foi horrível"

E a IA irá:

* Classificar o sentimento
* Mostrar a confiança (%)
* Salvar no histórico (estilo chat)

---

## 🧠 Funcionalidades de IA

* Classificação de sentimentos com Machine Learning
* Modelo Naive Bayes treinado com dados personalizados
* Vetorização de texto com NLP (CountVectorizer)
* Exibição da confiança do modelo em tempo real
* Processamento de linguagem natural (NLP básico)

---

## 🎨 Funcionalidades de Interface (UX/UI)

* Interface moderna e responsiva
* Feedback visual por cor:

  * 🟢 Verde → Positivo
  * 🔴 Vermelho → Negativo
* Loading de análise da IA
* Histórico das análises (estilo ChatGPT)
* Botão para limpar histórico
* Estado inicial inteligente ("Teste Agora!")
* Design com foco em experiência do usuário

---

## 🛠️ Tecnologias Utilizadas

### Backend

* Python 3
* Flask
* Scikit-learn (Machine Learning)

### Frontend

* HTML5
* CSS3 (UI Responsiva)
* JavaScript
* Lucide Icons (ícones modernos)

### Inteligência Artificial

* Naive Bayes (MultinomialNB)
* NLP com CountVectorizer
* Análise de sentimento baseada em texto

---

## 📂 Estrutura do Projeto

```
ai-sentiment-analyzer/
│
├── app.py
├── README.md
├── requirements.txt
└── templates/
    └── index.html
```

---

## ⚙️ Como Rodar o Projeto Localmente

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/SEU-USUARIO/ai-sentiment-analyzer.git
```

### 2️⃣ Entrar na pasta do projeto

```bash
cd ai-sentiment-analyzer
```

### 3️⃣ Instalar as dependências

```bash
pip install flask scikit-learn
```

### 4️⃣ Executar o servidor

```bash
python app.py
```

### 5️⃣ Abrir no navegador

```
http://127.0.0.1:5000
```

---

## 📊 Como a IA Funciona (Explicação Técnica)

1. O usuário digita uma frase
2. O texto é transformado em vetores numéricos (NLP)
3. O modelo de Machine Learning analisa o padrão
4. A IA classifica como Positivo ou Negativo
5. O sistema calcula a probabilidade (confiança)
6. O resultado é exibido na interface + salvo no histórico

---

## 🧪 Exemplo de Uso

Entrada:

```
"Hoje foi um dia maravilhoso"
```

Saída:

```
Sentimento: Positivo
Confiança da IA: 92%
```

---

## 📈 Diferenciais do Projeto (Nível Estágio em IA + Dev)

* Integração completa entre IA + Web
* Arquitetura backend com Flask
* Aplicação real de Machine Learning
* Interface profissional e responsiva
* Gerenciamento de estado (histórico em memória)
* Experiência de usuário inspirada em aplicações de IA modernas

---

## 🎯 Objetivo do Projeto

Este projeto foi desenvolvido com foco em:

* Aprendizado de Inteligência Artificial aplicada
* Desenvolvimento Web com Python
* Portfólio para vagas de Estágio em Desenvolvimento e IA

---

## 👨‍💻 Autor

Desenvolvido por: **Caio Manfio**
Área: Desenvolvimento de Sistemas (ADS)
Foco: Inteligência Artificial + Desenvolvimento Web

---

## ⭐ Melhorias Futuras (Roadmap)

* Deploy online (Render ou Vercel)
* Banco de dados (SQLite/PostgreSQL)
* Suporte a mais idiomas
* Modelo de IA mais avançado (Deep Learning)
* API pública de análise de sentimentos
* Autenticação de usuários

---

## 📜 Licença

Este projeto é de uso educacional e para portfólio profissional.