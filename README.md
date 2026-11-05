# VanishCam — efeitos de desaparecimento para webcam

O vídeo é processado no seu computador. Ele não é gravado nem enviado a um servidor. A prévia e a câmera virtual exibem somente o vídeo e os efeitos, sem menus, instruções ou textos sobre a imagem.

## Apoie o projeto

Se gostar do projeto, você pode contribuir pelo [PayPal](https://paypal.me/nicovic17) ou pelo [Revolut](https://revolut.me/nicoauriemma).

## Baixar para Windows

[Baixe o VanishCam.exe](https://nicoaauriemma.vercel.app/shop) e abra o arquivo. Não é necessário instalar o Python.

## Primeiros passos

1. Abra `AVVIA_LOCALE.bat`. Na primeira execução, ele cria o ambiente `.venv-local` e instala as dependências; a internet é necessária nessa etapa.
2. No menu, escolha **Vanish Normal**, **Substituição Naruto**, **Teletransporte Goku** ou **Goku + Naruto** com o mouse ou as teclas `1` a `4`.
3. Saia do enquadramento por 4 segundos enquanto o fundo é capturado. O terminal avisa quando você pode voltar.

**Vanish Normal:** faça o sinal de V com os dedos indicador e médio. O desaparecimento começa quando o gesto é reconhecido e dura cerca de 3,6 segundos.

**Substituição Naruto:** junte as palmas diante da webcam, com os dedos lado a lado, sem entrelaçá-los. Mantenha as duas mãos visíveis. Quando o gesto for reconhecido, você desaparece em uma nuvem de fumaça e um tronco aparece, cai e fica na cena até você pressionar `R`. Não é necessário usar microfone. O modo usa `hand_landmarker.task` para acompanhar as duas mãos. O arquivo `assets/vanish.MP3` toca nos alto-falantes quando o desaparecimento começa, inclusive quando você usa `V` para ativá-lo. Você pode repetir o efeito depois de pressionar `R`.

**Teletransporte Goku:** mantenha os dedos indicador e médio juntos contra a testa por cerca de um segundo. A pessoa permanece na mesma posição enquanto riscos azuis passam pela frente e pelas laterais, acelerando antes do desaparecimento. Repita o gesto para voltar com o efeito invertido. O modo usa `hand_landmarker.task` e `face_landmarker.task` para localizar os dedos e a testa. O arquivo `assets/goku-teleport-sound.mp3` toca nas duas transições; caso não esteja presente, o app gera um som de teletransporte. A contagem regressiva aparece somente no terminal.

**Goku + Naruto:** os gestos de testa do Goku e de palmas unidas do Naruto ficam ativos na mesma sessão.

Para iniciar diretamente em um modo, sem abrir o menu:

```powershell
AVVIA_LOCALE.bat --mode normal
AVVIA_LOCALE.bat --mode naruto
AVVIA_LOCALE.bat --mode goku
AVVIA_LOCALE.bat --mode combo
```

Use `--effects` para combinar os efeitos:

```powershell
AVVIA_LOCALE.bat --effects naruto,goku
AVVIA_LOCALE.bat --effects normal,naruto,goku
```

Os modelos do MediaPipe na pasta `models/` são baixados automaticamente na primeira execução. Para verificar os modelos de todos os modos, os recursos incluídos e a decodificação de áudio sem abrir a webcam:

```powershell
.venv-local/Scripts/python.exe vanish_cam.py --self-test
```

Para processar uma imagem estática sem webcam, use o teste de imagem. Ele usa a ilustração de exemplo, verifica se a segmentação separa a pessoa do fundo e salva uma prévia lado a lado em `test-output/`:

```powershell
.venv-local/Scripts/python.exe vanish_cam.py --test-image
```

Você também pode informar uma foto própria com `--test-image caminho/para/foto.jpg` e escolher o destino da prévia com `--test-output caminho/para/previa.png`. A imagem incluída é uma amostra sintética para verificar a segmentação. O reconhecimento do gesto aparece como diagnóstico, mas a ilustração não garante um resultado realista para os gestos.

## Câmera virtual e OBS

Se o [OBS Studio](https://obsproject.com/) estiver instalado, o app tenta iniciar automaticamente a **OBS Virtual Camera**. O terminal mostra o nome do dispositivo. Use `AVVIA_VIRTUAL.bat` para exigir uma câmera virtual. No aplicativo que receber o vídeo, selecione **OBS Virtual Camera**. A câmera virtual do OBS precisa estar parada enquanto o VanishCam estiver transmitindo a saída.

Para usar o efeito em uma cena do OBS, execute `AVVIA_OBS.bat`, adicione uma fonte de **Captura de janela**, selecione **Webcam - Prévia** e inicie a câmera virtual no OBS. Como alternativa, use o [Unity Capture](https://github.com/schellingb/UnityCapture) como uma fonte de vídeo virtual separada:

```powershell
AVVIA_VIRTUAL.bat --virtual-backend unitycapture
```

O OBS oferece uma única câmera virtual. Não use a mesma câmera virtual ao mesmo tempo como entrada e saída do OBS. Consulte a [documentação do pyvirtualcam](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controles

- `ESPAÇO`: captura um novo fundo.
- `R`: redefine o efeito e faz a pessoa reaparecer.
- `V`: ativa manualmente o efeito. Em sessões combinadas, cada toque percorre os efeitos ativos.
- `F`: alterna entre tela cheia e janela.
- `Q` ou `ESC`: fecha o app e libera as câmeras.

Para alterar a câmera, a resolução e o limite máximo de FPS da saída virtual:

```powershell
AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

Para diagnosticar o reconhecimento de gestos sem adicionar texto ao vídeo, use `--gesture-debug`. As informações aparecem somente no terminal.

## Executar a partir do código-fonte

`AVVIA_LOCALE.bat` procura o Python 3.10 ou mais recente, dando preferência ao 3.12 e ao 3.13, e cria o ambiente local automaticamente. No Python 3.14, o script instala `pygame-ce`, que fornece o mesmo módulo `pygame` usado pelo app.

Para instalar as dependências manualmente:

```powershell
py -3 -m venv .venv-local
.venv-local/Scripts/python.exe -m pip install -r requirements.txt
.venv-local/Scripts/python.exe vanish_cam.py
```

Para executar os testes automatizados:

```powershell
.venv-local/Scripts/python.exe -m unittest discover -p "test_*.py" -v
```

## Protótipo legado

`run3.py` é o protótipo original de rastreamento de gestos e memes. Ele é separado do VanishCam, não é usado pelos inicializadores nem pela compilação para Windows, e suas imagens de memes não estão incluídas.

## Recursos visuais

O [tronco de substituição](assets/substitution_log.png) é uma ilustração original criada com o ImageGen usando a imagem fornecida como referência visual. A fumaça é desenhada em tempo real por `naruto_effect.py` para acompanhar a posição da pessoa e se ajustar ao vídeo. Também avaliamos uma folha de sprites de fumaça com licença CC0, mas ela não foi incluída porque a animação procedural se adapta melhor às mudanças de posição. Os rastros verticais do Goku são calculados em tempo real a partir da silhueta. Se `assets/goku-teleport-sound.mp3` não estiver presente, `vanish_sound.py` sintetiza um WAV de fallback. O projeto não inclui cenas ou imagens de animes.
