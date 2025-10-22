import os
import sys
import logging
from datetime import datetime

class LogManager:
    def __init__(self, subpasta='main'):
        self.subpasta = subpasta
        self.logger = None
        self.caminho_log = self._criar_caminhos_logs()
        self._configurar_logger()

    def _criar_pasta_logs(self):
        if getattr(sys, 'frozen', False):
            raiz_projeto = os.path.dirname(sys.executable)
        else:
            raiz_projeto = os.path.dirname(os.path.abspath(__file__))
        
        base_dir = os.path.join(raiz_projeto, 'logs')
        caminho_completo = os.path.join(base_dir, self.subpasta)

        os.makedirs(caminho_completo, exist_ok=True)
        return caminho_completo

    def _criar_caminhos_logs(self):
        pasta_logs = self._criar_pasta_logs()
        data_hoje = datetime.now().strftime('%Y-%m-%d')
        caminho_log = os.path.join(pasta_logs, f"{data_hoje}.log")
        return caminho_log

    def _configurar_logger(self):
        self.logger = logging.getLogger(f'logger_{self.subpasta}')
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            formato = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler = logging.FileHandler(self.caminho_log, encoding='utf-8')
            handler.setFormatter(formato)
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger

    def flush_and_close(self):
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
