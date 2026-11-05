# Recursos visuais e áudio

## Tronco de substituição

`substitution_log.png` foi criado com o ImageGen. A imagem enviada pelo usuário foi usada somente como referência de formato e atmosfera, não como alvo de edição. O sprite é uma ilustração original com fundo transparente.

Descrição usada na geração:

> Crie um sprite 2D ORIGINAL, pintado à mão e em estilo de anime, para composição em vídeo ao vivo. Mostre um único tronco curto e grosso de substituição, flutuando na diagonal, do canto inferior esquerdo ao superior direito. Use casca verde-sálvia clara e dessaturada, com veios de madeira, rachaduras escuras e pequenos nós. Mostre a seção circular cortada na extremidade inferior esquerda, com anéis de madeira concêntricos. Silhueta nítida, cel shading, realces e acabamento de animação japonesa. Centralize o tronco e deixe uma margem transparente discreta. Sem fumaça, partículas, fundo, cenário, sombra externa, pessoas, texto, símbolos ou logotipos. Fundo realmente transparente; recorte isolado, pronto para ser usado sobre o vídeo da webcam.

O app cria a fumaça em tempo real com formas procedurais em `naruto_effect.py`. Uma folha de sprites de fumaça com licença CC0 foi avaliada, mas não foi incluída porque a fumaça procedural acompanha a posição detectada da pessoa.

## Som do teletransporte

`goku-teleport-sound.mp3` é o áudio usado nas duas transições do modo Goku. Se o arquivo não estiver presente, `vanish_sound.py` gera um WAV curto (`vanish_teleport.wav`) na pasta temporária do sistema.
