# RoboCOP — teste direto no Google Colab

Este ramo testa a arquitetura **estado resolutivo -> memória de estados futuros -> controle físico local**, sem BahiaRT e sem Docker.

## Uma única célula no Colab

Copie e execute:

```python
!pip -q install "gymnasium[mujoco]" matplotlib
!git clone -q -b migration/bahiart-mujoco https://github.com/marceloroldao/robocop.git
%cd /content/robocop
%run colab/robocop_resolutive_centroidal_v1.py
```

Se `/content/robocop` já existir em uma sessão reutilizada:

```python
%cd /content/robocop
!git pull
%run colab/robocop_resolutive_centroidal_v1.py
```

## Saída esperada

O script imprime `steps`, `memory`, `learned`, `recalls` e `recall_rate`, e desenha gráficos de altura, coerência/estabilidade e crescimento da memória.

O V1 é deliberadamente um teste mínimo. Ele não pretende demonstrar marcha autônoma; testa se uma memória resolutiva não neural consegue aprender e recuperar regiões corporais futuras úteis no Humanoid-v5 antes da introdução de fase corporal inferida, contatos explícitos e WBC/MPC.
