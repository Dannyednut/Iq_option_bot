# config_manager.py
import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_sections = ['credentials', 'channels', 'trading', 'risk_management']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate credentials
        required_credentials = ['email', 'password', 'telegram_token']
        for cred in required_credentials:
            if cred not in self.config['credentials']:
                raise ValueError(f"Missing required credential: {cred}")

    def get_credentials(self) -> Dict[str, str]:
        """Get API credentials."""
        return self.config['credentials']

    def get_channels(self) -> list:
        """Get telegram channel IDs."""
        return self.config['channels']

    def get_trading_settings(self) -> Dict[str, Any]:
        """Get trading settings."""
        return self.config['trading']

    def get_risk_settings(self) -> Dict[str, float]:
        """Get risk management settings."""
        return self.config['risk_management']

    def get_gale_settings(self) -> Dict[str, Any]:
        """Get gale strategy settings."""
        return self.config['gale_settings']

    def get_indicator_settings(self) -> Dict[str, Dict]:
        """Get technical indicator settings."""
        return self.config['indicators']

    def get_message_templates(self) -> Dict[str, str]:
        """Get message templates for notifications."""
        return self.config['telegram_messages']