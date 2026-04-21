from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Base is the single source of truth — always imported from the top-level module
# so all models register on the same metadata object.
from database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatSession(Base):
    __tablename__ = "chatsession"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.timestamp",
    )

    __table_args__ = (Index("ix_chatsession_session_id", "session_id"),)


class ChatMessage(Base):
    __tablename__ = "chatmessage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chatsession.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    # "user" | "assistant"
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    __table_args__ = (Index("ix_chatmessage_session_id", "session_id"),)


class AnalysisLog(Base):
    __tablename__ = "analysislog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Null means analysis returned no disease identification
    disease_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Full JSON response stored as a string for flexibility — parse at read time
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    is_diseased: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


class CommunityReport(Base):
    """
    Anonymous community disease report submitted by a farmer.

    No user identity or photos are stored — just the crop/disease/location
    so other farmers can see what is spreading in their area.
    """

    __tablename__ = "communityreport"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    disease_name:  Mapped[str] = mapped_column(String(128), nullable=False)
    crop_name:     Mapped[str] = mapped_column(String(64),  nullable=False)
    location_name: Mapped[str] = mapped_column(String(128), nullable=False)

    division: Mapped[str | None] = mapped_column(String(64),  nullable=True)
    severity: Mapped[str]        = mapped_column(String(16),  default="Medium", nullable=False)
    notes:    Mapped[str | None] = mapped_column(String(200), nullable=True)

    is_anonymous: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    # Redundant with created_at but easier to query/filter by calendar day.
    report_date: Mapped[str] = mapped_column(String(10), nullable=False)

    __table_args__ = (
        Index("ix_communityreport_created_at", "created_at"),
        Index("ix_communityreport_division",   "division"),
        Index("ix_communityreport_crop_name",  "crop_name"),
    )
