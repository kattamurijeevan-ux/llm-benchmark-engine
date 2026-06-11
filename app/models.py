from pydantic import BaseModel

class BenchmarkRunRequest(BaseModel):
    model: str
    domain: str
    num_samples: int = 20

class CompareRequest(BaseModel):
    models: list[str]
    domain: str
    num_samples: int = 20