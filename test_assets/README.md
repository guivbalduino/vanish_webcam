# Imagem de teste estático

`sample_person.png` é uma cena sintética e determinística, gerada por `generate_sample_image.py`. Ela é incluída no executável do Windows e usada por `vanish_cam.py --test-image` para executar a segmentação do MediaPipe sem abrir a webcam. A prévia mostra a imagem original, a pessoa recortada e a máscara de segmentação.

Esta amostra verifica se a inferência é executada e separa primeiro plano e fundo. Ela não é uma imagem de referência para reconhecimento de gestos; para inspecionar esse diagnóstico, use uma foto real com a mão e a pose bem visíveis.

Para gerar novamente a imagem:

```powershell
.venv-local/Scripts/python.exe test_assets/generate_sample_image.py
```