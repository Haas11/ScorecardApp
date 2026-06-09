from __future__ import annotations

import warnings
from typing import Optional
from pydantic import BaseModel, model_validator


class GameInfo(BaseModel):
    teams: dict[str, str]
    date: Optional[str] = None
    game_number: Optional[str] = None


class PASummary(BaseModel):
    PA: Optional[int] = None
    AB: Optional[int] = None
    R: Optional[int] = None
    H: Optional[int] = None
    model_config = {"populate_by_name": True}

    # Extra fields via __pydantic_extra__ not needed; use explicit optional fields
    two_B: Optional[int] = None
    three_B: Optional[int] = None
    HR: Optional[int] = None
    BB: Optional[int] = None
    HP: Optional[int] = None
    K: Optional[int] = None
    SB: Optional[int] = None
    CS: Optional[int] = None
    RBI: Optional[int] = None
    SAC: Optional[int] = None
    SF: Optional[int] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class PlateAppearance(BaseModel):
    inning: int
    result: str
    run_scored: bool = False
    rbi: int = 0
    sb: int = 0
    cs: int = 0
    notes: str = ""
    confidence: str = "high"


class PlayerEntry(BaseModel):
    name: str
    position: Optional[int] = None
    jersey_number: Optional[int] = None
    innings_played: Optional[str] = None
    plate_appearances: list[PlateAppearance] = []
    summary: Optional[PASummary] = None

    @model_validator(mode="after")
    def check_pa_count(self) -> "PlayerEntry":
        if self.summary and self.summary.PA is not None:
            if len(self.plate_appearances) > self.summary.PA:
                warnings.warn(
                    f"Player '{self.name}': recorded {len(self.plate_appearances)} PAs "
                    f"but summary.PA={self.summary.PA}"
                )
        return self


class LineupSlot(BaseModel):
    batting_order: int
    players: list[PlayerEntry]


class PitchingLine(BaseModel):
    name: Optional[str] = None
    innings_pitched: Optional[float] = None
    runs_allowed: Optional[int] = None
    earned_runs: Optional[int] = None
    strikeouts: Optional[int] = None
    walks: Optional[int] = None
    confidence: str = "high"


class AmbiguousCell(BaseModel):
    batter: str
    inning: int
    raw_text: str


class InningTotals(BaseModel):
    runs_per_inning: list[int] = []
    errors_total: Optional[int] = None


class GameExtraction(BaseModel):
    game: GameInfo
    lineup: list[LineupSlot]
    pitching: list[PitchingLine] = []
    inning_totals: Optional[InningTotals] = None
    ambiguous_cells: list[AmbiguousCell] = []
