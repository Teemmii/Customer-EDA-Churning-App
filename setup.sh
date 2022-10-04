mkdir -p ~/ .streamlit2/

echo"\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/ .streamlit2/config.toml
