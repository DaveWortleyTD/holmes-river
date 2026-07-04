from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    spypoint_username: str
    spypoint_password: str
    spypoint_camera_id: str = ""

    anthropic_api_key: str = ""
    poll_interval_minutes: int = 30
    db_path: str = "./readings.db"
    gauge_line_spacing_cm: int = 20
    gauge_reference_image: str = ""  # path to a clear reference photo of the full gauge staff


settings = Settings()
