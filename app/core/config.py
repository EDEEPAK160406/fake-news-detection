from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multimodal Fake News Detector"
    app_version: str = "1.0.0"
    model_path: str = "artifacts/multimodal_model.joblib"
    vectorizer_path: str = "artifacts/text_vectorizer.joblib"
    image_auth_model_path: str = "artifacts/image_authenticity_model.joblib"
    image_auth_dataset_dir: str = "data/image_authenticity"
    image_auth_cnn_path: str = "artifacts/image_authenticity_cnn.pt"
    video_auth_model_path: str = "artifacts/video_multimodal.pt"
    video_auth_cnn_path: str = "artifacts/video_authenticity_cnn.pt"
    video_auth_dataset_dir: str = "data/video_authenticity"
    video_frame_sample_rate: int = 2  # sample one frame every N frames by default
    video_model_fake_threshold: float = 0.45
    video_fake_threshold: float = 0.66
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "fake_news_system"
    trusted_domains: str = "reuters.com,apnews.com,bbc.com,nytimes.com"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
