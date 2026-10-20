from app.critics.accuracy import AccuracyCritic
from app.critics.completeness import CompletenessCritic
from app.critics.consistency import ConsistencyCritic
from app.critics.grounding import GroundingCritic

ALL_CRITICS = [AccuracyCritic, ConsistencyCritic, CompletenessCritic, GroundingCritic]
