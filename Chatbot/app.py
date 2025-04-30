import streamlit as st
import time

# Configuração da página
st.set_page_config(
    page_title="Chatbot de Músicas - TZ da Coronel",
    page_icon="🎤",
    layout="centered"
)

# Personalização de estilo
st.markdown("""
    <style>
    .main {
        text-align: center;
    }
    .title {
        color: #f1c40f;
        font-size: 3em;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.2em;
        color: #16a085;
        font-style: italic;
    }
    .chatbox {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 20px;
        max-height: 400px;
        overflow-y: scroll;
        margin-bottom: 20px;
        background: #e9f7f7;
    }
    .user-message {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 15px;
        max-width: 75%;
        margin: 5px 0;
        text-align: left;
        margin-left: 10%;
    }
    .bot-message {
        background-color: #2980b9;
        color: white;
        padding: 10px;
        border-radius: 15px;
        max-width: 75%;
        margin: 5px 0;
        text-align: left;
        margin-right: 10%;
    }
    .input-box {
        background-color: #ffffff;
        border: 2px solid #2980b9;
        border-radius: 20px;
        padding: 12px 20px;
        width: 100%;
        max-width: 700px;
        box-sizing: border-box;
        margin-bottom: 20px;
        font-size: 16px;
    }
    .input-box:focus {
        border-color: #f39c12;
        outline: none;
    }
    .album-image {
        width: 100%;
        height: auto;
        border-radius: 10px;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #888;
        margin-top: 20px;
    }
    .clear-btn {
        background-color: #e74c3c;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .refresh-btn {
        background-color: #f39c12;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Título da página
st.markdown("<div class='title'>🎤 Chatbot de Músicas - TZ da Coronel</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Pergunte sobre TZ da Coronel, suas faixas e álbuns!</div>", unsafe_allow_html=True)

# Dicionário com informações sobre músicas, álbuns e o artista
musicas_db = {
    "Acordado eu Sonho": {
        "artista": "TZ da Coronel",
        "link": "https://www.youtube.com/watch?v=Fx24q9OFSn8",
        "faixa": "Acordado eu Sonho",
        "ano": 2024,
        "album": "Direto da Selva",
        "estilo": "Rap, Trap"
    },
    "A Grana Vindo": {
        "artista": "TZ da Coronel",
        "link": "https://www.youtube.com/watch?v=IDIJ14Owj8g",
        "faixa": "A Grana Vindo",
        "ano": 2024,
        "album": "Direto da Selva",
        "estilo": "Rap, Trap"
    },
    "Qual é seu Desejo?": {
        "artista": "TZ da Coronel",
        "link": "https://www.youtube.com/watch?v=5Yq5J0d2g8w",
        "faixa": "Qual é seu Desejo?",
        "ano": 2024,
        "album": "Direto da Selva",
        "estilo": "Rap, Trap"
    },
    "Vem ou Não Vem": {
        "artista": "TZ da Coronel",
        "link": "https://www.youtube.com/watch?v=3XqF0gZpQdE",
        "faixa": "Vem ou Não Vem",
        "ano": 2024,
        "album": "Direto da Selva",
        "estilo": "Rap, Trap"
    },
    "Direto da Selva": {
        "artista": "TZ da Coronel",
        "link": "https://www.youtube.com/watch?v=4J6g0g8Wq9A",
        "faixa": "Direto da Selva",
        "ano": 2024,
        "album": "Direto da Selva",
        "estilo": "Rap, Trap"
    },
    # Adicione mais faixas e informações aqui
}

# Função para gerar resposta do chatbot
def bot_resposta(pergunta):
    pergunta = pergunta.lower()

    # Respostas sobre faixas e músicas
    if "faixa" in pergunta or "música" in pergunta:
        for faixa, info in musicas_db.items():
            if faixa.lower() in pergunta or any(word in pergunta for word in faixa.lower().split()):
                return f"A música *{faixa}* é do artista *{info['artista']}*, do álbum *{info['album']}* de {info['ano']}. Você pode ouvir aqui: {info['link']}"
        return "Desculpe, não encontrei essa música. Tente outra!"
    
    # Respostas sobre o artista TZ da Coronel
    elif "artista" in pergunta or "tz da coronel" in pergunta:
        return ("TZ da Coronel é um artista de rap e trap, conhecido por suas faixas com letras intensas, "
                "que falam sobre superação, a vida nas ruas e a busca por liberdade financeira. Seu álbum mais recente "
                "é 'Direto da Selva', lançado em 2024.")

    # Respostas sobre o estilo musical
    elif "estilo" in pergunta or "gênero" in pergunta:
        return "O estilo musical de TZ da Coronel é rap e trap, com letras que abordam temas como a vida nas ruas, ambição, e a busca pela liberdade financeira."

    # Respostas sobre álbuns
    elif "álbum" in pergunta or "disco" in pergunta:
        return "O álbum mais recente de TZ da Coronel é 'Direto da Selva', lançado em 2024. O álbum apresenta faixas como 'Acordado eu Sonho', 'Vem ou Não Vem', entre outras."

    # Respostas sobre os shows
    elif "show" in pergunta or "próximo show" in pergunta:
        return "Atualmente, não há informações sobre os próximos shows de TZ da Coronel. Fique ligado nas redes sociais do artista para atualizações!"

    # Caso o bot não consiga identificar
    else:
        return "Eu só posso responder sobre músicas e artistas. Pergunte algo relacionado a TZ da Coronel, suas faixas ou estilo musical!"

# Caixa de entrada para o usuário
usuario_input = st.text_input("💬 Pergunte sobre TZ da Coronel:", "")

# Botão para limpar o chat
if st.button("Limpar Chat", key="clear_chat"):
    st.session_state.messages = []  # Limpar o histórico de mensagens

# Botão para reiniciar o chat
if st.button("Reiniciar Chat", key="refresh_chat"):
    st.session_state.messages = []  # Limpar o histórico de mensagens e reiniciar
    st.experimental_rerun()  # Recarregar a página para reiniciar o chat

# Armazenar o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico de mensagens
if st.session_state.messages:
    with st.container():
        for msg in st.session_state.messages:
            if msg["sender"] == "user":
                st.markdown(f'<div class="user-message">Você: {msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">Bot: {msg["text"]}</div>', unsafe_allow_html=True)

# Processar o input do usuário
if usuario_input:
    # Adicionar mensagem do usuário no histórico
    st.session_state.messages.append({"sender": "user", "text": usuario_input})
    
    # Gerar a resposta do bot
    resposta = bot_resposta(usuario_input)
    
    # Adicionar a resposta do bot no histórico
    st.session_state.messages.append({"sender": "bot", "text": resposta})

    # Limpar a caixa de texto após o envio
    st.text_input("💬 Pergunte sobre TZ da Coronel:", "", key="clear", disabled=True)

# Rodapé
st.markdown("---")
st.markdown("<p style='text-align: center;'>🛠️ Desenvolvido com ❤️ usando Streamlit | Chatbot Musical - TZ da Coronel | 2024</p>", unsafe_allow_html=True)
