"""Configuração comum dos testes.

No Mac, o LightGBM (libomp do Homebrew) e o PyTorch (libomp próprio) no mesmo processo derrubam o
Python com "Segmentation fault" quando o LightGBM é carregado primeiro. Importar o torch antes de
qualquer teste fixa a ordem segura. Sem torch instalado, nada muda.
"""
try:
    import torch  # noqa: F401
except ImportError:
    pass
