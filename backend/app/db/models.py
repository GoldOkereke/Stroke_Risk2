# backend/db/models.py - SQLAlchemy ORM models: how data is stored.
#
# Contrast with app/models/schema.py: that file validates data crossing the API
# boundary (Pydantic); this one maps Python classes to database tables (ORM).
# The same entity can appear in both, for a different purpose.

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, JSON, String
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    # Inheriting from `Base` registers the class with SQLAlchemy's metadata,
    # which is what lets `Base.metadata.create_all(engine)` build the table.
    # Without `__tablename__`, SQLAlchemy would derive "user" from the class
    # name, which also clashes with SQL's USER keyword.
    __tablename__ = "users"

    # A string id instead of an auto-incrementing integer: ids stay portable
    # when they are generated outside this database later.
    id = Column(String, primary_key=True, index=True)

    # One account per address. `unique=True` enforces that in the database,
    # `nullable=False` makes the email mandatory, and the index speeds up the
    # by-email lookup performed at login.
    email = Column(String, unique=True, nullable=False, index=True)

    # Only the hash is stored, never the password itself; hashing happens in
    # the security layer before a value reaches the database.
    hashed_password = Column(String, nullable=False)

    # Supplied once, when the row is inserted. UTC keeps timestamps comparable
    # regardless of where the server or the user is.
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Object-level access to the child rows (`user.test_results`). The database
    # link itself is a ForeignKey on those child tables, because each child
    # stores the id of the user it belongs to - so it cannot live here.
    #
    # The target is a string and is resolved lazily, so the child classes only
    # have to exist when the mappers are configured, not when this module is
    # imported. `back_populates` names the matching attribute on the child.
    baseline_samples = relationship(
        "BaselineSample",
        back_populates="user",
    )

    test_results = relationship(
        "TestResult",
        back_populates="user",
    )


class BaselineSample(Base):
    # A user's personal reference measurement, not a diagnosis. Future readings
    # are compared against it, so a 105 BPM result means more when the same
    # user's resting baseline is 65 BPM than when only a population threshold
    # is applied. A baseline may come from any supported modality (ECG, PPG,
    # and others as the system grows).
    __tablename__ = "baseline_samples"

    # Its own id, separate from the owner's: one user can hold several baseline
    # samples, and each must be addressable independently of who owns it.
    id = Column(String, primary_key=True, index=True)

    # The owner. Same type as `users.id`, and `nullable=False` means a baseline
    # can never exist without a user. `ForeignKey` is what makes the database
    # itself enforce the link, not just the Python code.
    user_id = Column(
        String,
        ForeignKey("users.id"),
        nullable=False,
    )

    # What kind of measurement this sample holds, e.g. "ECG" or "PPG". A new
    # ECG recording must be compared with the user's ECG baseline, not their
    # PPG one, so the lookup is (user_id, modality).
    #
    # A plain string rather than an enum: the supported modalities are still
    # being defined, and adding a value should not require a schema migration.
    # Validation belongs in the application layer for now.
    modality = Column(String, nullable=False)

    # The extracted measurements as a JSON object, so an ECG baseline and a PPG
    # baseline can share one column despite different shapes:
    #   ECG: {"mean_heart_rate": 68.5, "mean_rr_interval": 0.88, ...}
    #   PPG: {"mean_pulse_rate": 70.2, "mean_pulse_interval": 0.86, ...}
    # This avoids a column per measurement, or a table per modality.
    #
    # It stores extracted features, not the raw waveform; raw signals travel
    # through the ingestion and storage pipeline instead. Because the content is
    # free-form, shape validation belongs in the application layer.
    measurements = Column(JSON, nullable=False)

    # When the sample was collected. A personal baseline is not permanent -
    # physiology changes - so this is what makes "the user's most recent ECG
    # baseline" and trend comparisons across periods possible.
    #
    # A Python-side `default`, so it is applied when a row is inserted rather
    # than by the database. UTC keeps timestamps comparable across locations.
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Python-side navigation of that same link: `baseline.user` returns the
    # owner, and `user.baseline_samples` returns the collection. The foreign key
    # enforces the connection in the database; `relationship()` only makes it
    # convenient to traverse from code.
    user = relationship(
        "User",
        back_populates="baseline_samples",
    )


class TestResult(Base):
    # One completed assessment, for any modality: the record written after a
    # signal is processed and the model returns its prediction. It stores the
    # outcome, not the uploaded signal itself.
    __tablename__ = "test_results"

    # A user may accumulate hundreds of assessments, and each must be
    # addressable on its own.
    id = Column(String, primary_key=True, index=True)

    # The account that submitted the assessment. This is what keeps one user's
    # history separate from another's.
    user_id = Column(
        String,
        ForeignKey("users.id"),
        nullable=False,
    )

    # Which assessment was performed, e.g. "ECG", "PPG", "FACE", "DYSARTHRIA",
    # "PARKINSONS". Without it, results from different modalities are
    # indistinguishable in this table.
    #
    # Deliberately the same vocabulary as `BaselineSample.modality`, so a result
    # can be paired with the matching baseline (test_type == modality). A plain
    # string for the same reason as `modality`: the supported types are still
    # evolving, and validation belongs in the application layer.
    test_type = Column(String, nullable=False)

    # The numerical output of the assessment: an AFib probability, a PPG
    # positive-class probability, or a combined score from the fusion engine.
    # `Float` because these are decimal fractions such as 0.82 or 0.947.
    #
    # Nullable on purpose: not every assessment yields one comparable number.
    # The Parkinson's voice model predicts UPDRS regression targets, and an AFib
    # probability is not a validated stroke probability. `risk_score` is
    # reserved for scores whose meaning the pipeline defines explicitly; raw
    # model outputs stay in the structured result data.
    risk_score = Column(Float, nullable=True)

    # The categorical reading of that score - LOW / MEDIUM / HIGH / CRITICAL -
    # stored rather than derived, so a displayed assessment keeps the level it
    # was assigned. Score and level are separate because the number and its
    # interpretation are decided by the assessment pipeline, not by the schema:
    # 0.82 -> HIGH, 0.45 -> MEDIUM. Nullable for assessments that produce no
    # category at all.
    risk_level = Column(String, nullable=True)

    # Quality of the submitted signal: POOR / FAIR / GOOD / EXCELLENT, matching
    # the `SignalQuality` enum in the Pydantic layer. Stored so a result can be
    # judged against how trustworthy its input was.
    #
    # Nullable because not every assessment has a signal in this sense - a facial
    # image assessment has no ECG-style quality rating - and because quality may
    # not always be computable.
    signal_quality = Column(String, nullable=True)

    # Everything else the pipeline produced, as a JSON object: predicted class,
    # model confidence, extracted features, and any modality-specific extras.
    #   ECG: {"prediction": "AFIB", "confidence": 0.82, "mean_heart_rate": 68.5}
    # Because each modality returns a different shape, JSON avoids a column per
    # output, and keeps the raw model values that `risk_score` and `risk_level`
    # deliberately do not carry.
    #
    # Required: an assessment without any result payload is not a result. As
    # with `measurements`, the content is free-form, so shape validation belongs
    # in the application layer.
    result_data = Column(JSON, nullable=False)

    # When the assessment was performed. Ordering by this gives a chronological
    # history and makes "compare this result with earlier ones" possible.
    # Python-side default, applied at insert; UTC keeps timestamps comparable.
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # The mirror image of `User.test_results`: the foreign key enforces the link
    # in the database, `relationship()` exposes it in both directions
    # (`user.test_results` and `test_result.user`).
    user = relationship(
        "User",
        back_populates="test_results",
    )
