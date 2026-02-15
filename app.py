# Importa ferramentas do Flask para criar o site
from flask import Flask, render_template, request, redirect, url_for
# Importa ferramenta para transformar texto em números
from sklearn.feature_extraction.text import CountVectorizer
# Importa o algoritmo de Inteligência Artificial (Naive Bayes)
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)  # Inicializa a aplicação Flask

# Histórico em memória (estilo chat)
historico = []  # Cria uma lista vazia para guardar o histórico de mensagens analisadas

# Base de treino
frases_positivas = [  # Lista de exemplos de frases positivas para ensinar a IA
    "estou feliz", "que dia maravilhoso", "estou animado", "isso é incrível",
    "estou muito bem", "hoje foi um ótimo dia", "isso é muito bom",
    "estou motivado", "que coisa excelente", "isso é fantástico", "oi!", "bom dia!", "boa tarde!", "boa noite!", "obrigado!", "parabéns!", "feliz aniversário!", "meu amor", "minha vida", "meu coração", "te amo", "você é incrível", "você é a melhor pessoa do mundo", "você é minha razão de viver", "você é minha felicidade", "você é minha luz", "você é minha inspiração", "você é minha alegria", "você é minha paz", "você é minha esperança", "você é minha força", "você é minha motivação", "você é minha razão de sorrir", "você é minha razão de ser feliz", "você é minha razão de amar", "você é minha razão de viver feliz", "eu amo você"
]

frases_negativas = [  # Lista de exemplos de frases negativas para ensinar a IA
    "estou triste", "que dia ruim", "estou cansado", "isso é horrível",
    "estou desanimado", "foi um péssimo dia", "isso é muito ruim",
    "estou frustrado", "que coisa horrível", "isso é horrível", "estou com raiva", "isso é terrível", "estou desapontado", "isso é uma droga", "estou deprimido", "isso é um desastre", "estou preocupado", "isso é um pesadelo", "estou com medo", "isso é assustador", "vai se ferrar", "isso é uma merda", "estou com ódio", "isso é um inferno", "estou com nojo", "isso é nojento", "estou com vergonha", "isso é embaraçoso", "estou com ciúmes", "isso é irritante"
]

# Junta todas as frases (positivas e negativas) em uma única lista
frases = frases_positivas + frases_negativas
# Cria os gabaritos: 1 para positivo, 0 para negativo
sentimentos = [1] * len(frases_positivas) + [0] * len(frases_negativas)

# Configura o conversor de texto (aceita palavras soltas e pares de palavras)
vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2))
# Converte as frases de treino em uma matriz de números que a IA entende
X = vectorizer.fit_transform(frases)

modelo = MultinomialNB()  # Cria uma instância do modelo Naive Bayes
# Treina o modelo usando os dados (X) e as respostas corretas (sentimentos)
modelo.fit(X, sentimentos)


# Define a rota principal do site, aceitando envio de dados (POST)
@app.route("/", methods=["GET", "POST"])
def index():  # Função executada ao acessar a página inicial
    resultado = None  # Variável para guardar se é Positivo ou Negativo
    # Variável para definir a cor do design (começa cinza/neutro)
    cor = "neutro"
    confianca = None  # Variável para guardar a certeza da IA (em %)
    mensagem = "Teste Agora!"  # Mensagem de título inicial

    if request.method == "POST":  # Verifica se o usuário clicou no botão de enviar
        # Pega o texto que o usuário digitou no formulário
        texto = request.form["texto"]

        # Converte o texto do usuário para números (igual ao treino)
        texto_transformado = vectorizer.transform([texto])
        # A IA faz a previsão: retorna 0 ou 1
        previsao = modelo.predict(texto_transformado)[0]
        # Calcula a probabilidade matemática de cada opção
        probabilidades = modelo.predict_proba(texto_transformado)[0]
        # Pega a maior probabilidade e transforma em porcentagem
        confianca = round(max(probabilidades) * 100, 2)

        if previsao == 1:  # Se a previsão for 1 (Positivo)
            resultado = "Positivo"  # Define o texto do resultado
            cor = "positivo"  # Define a classe CSS para verde
            mensagem = "Sentimento Positivo Detectado"  # Define a mensagem de feedback
        else:  # Se não for 1 (então é 0, Negativo)
            resultado = "Negativo"  # Define o texto do resultado
            cor = "negativo"  # Define a classe CSS para vermelho
            mensagem = "Sentimento Negativo Detectado"  # Define a mensagem de feedback

        # Adiciona no histórico (mais recente primeiro)
        historico.insert(0, {  # Insere um novo registro no topo da lista de histórico
            "texto": texto,  # Salva o texto original
            "resultado": resultado,  # Salva o resultado da IA
            "cor": cor,  # Salva a cor usada
            "confianca": confianca  # Salva a porcentagem de confiança
        })

    return render_template(  # Renderiza o arquivo HTML para o usuário ver
        "index.html",  # Nome do arquivo HTML
        resultado=resultado,  # Envia a variável resultado para o HTML
        cor=cor,  # Envia a variável cor
        confianca=confianca,  # Envia a confiança
        mensagem=mensagem,  # Envia a mensagem
        historico=historico  # Envia a lista completa do histórico
    )


@app.route("/limpar-historico")  # Rota específica para apagar o histórico
def limpar_historico():  # Função executada ao acessar essa rota
    historico.clear()  # Limpa todos os itens da lista da memória
    # Redireciona o usuário de volta para a página inicial
    return redirect(url_for("index"))


if __name__ == "__main__":  # Verifica se o arquivo está sendo executado diretamente
    app.run(debug=True)  # Inicia o servidor web em modo de depuração (debug)
