"""``configs/samples.yaml`` models: one entry per capture session (``config.load_samples``).

Piece chains (S08, user decision on open question 10): every session is its own piece unless its
entry says ``continues: <session_id>``. Then it inherits the parent's ``piece_id`` (CV groups stay
leakage-free) and its labels continue the parent's age (``labels.labels``). Marking a session as a
continuation is a YAML edit only; no code change.
Excluding a session (``exclude: true`` + ``exclude_reason``) keeps its raw data and features but
drops it from labels / evaluation with a logged reason (S08: exp3_D1A, user decision on question 9).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TempSpec(_Section):
    mode: Literal["weather", "fixed", "log"]
    value_c: float | None = None  # °C, used by mode 'fixed' (manual constant)
    log_file: str | None = None  # CSV timestamp,temp_c[,rh], used by mode 'log' (repo-relative)

    @model_validator(mode="after")
    def _mode_needs_value(self) -> TempSpec:
        if self.mode == "fixed" and self.value_c is None:
            raise ValueError("temp.mode 'fixed' needs temp.value_c (°C)")
        if self.mode == "log" and not self.log_file:
            raise ValueError("temp.mode 'log' needs temp.log_file")
        return self


class SessionSpec(_Section):
    experiment: str
    folder: str
    piece_id: str | None = None
    storage_day: int | None = None
    pre_age_h: float | None = Field(default=None, ge=0)  # h at room temp before filming
    continues: str | None = None  # session id this one continues (same physical piece)
    storage_temp_c: float | None = None  # °C between the parent's end and this start (fridge?)
    exclude: bool = False  # drop from labels / evaluation (data kept)
    exclude_reason: str = ""
    temp: TempSpec
    notes: str = ""

    @model_validator(mode="after")
    def _exclude_needs_reason(self) -> SessionSpec:
        if self.exclude and not self.exclude_reason:
            raise ValueError("exclude: true needs exclude_reason")
        return self


class SamplesConfig(_Section):
    sessions: dict[str, SessionSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _chains(self) -> SamplesConfig:
        for sid in self.sessions:
            seen, cur = {sid}, self.sessions[sid].continues
            while cur is not None:
                if cur not in self.sessions:
                    raise ValueError(f"{sid}: continues unknown session {cur!r}")
                if cur in seen:
                    raise ValueError(f"{sid}: 'continues' chain has a cycle")
                seen.add(cur)
                cur = self.sessions[cur].continues
        return self

    def chain(self, sid: str) -> list[str]:
        """Sessions of ``sid``'s piece from the first one to ``sid`` (``[sid]`` if standalone)."""
        out = [sid]
        while (spec := self.sessions.get(out[0])) is not None and spec.continues:
            out.insert(0, spec.continues)
        return out
