"""Fondations Voice Alert sans appels reels."""
from alembic import op
from voice_alert import AlertBase

revision = "20260828_01"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    AlertBase.metadata.create_all(bind=bind, checkfirst=True)

def downgrade():
    # Les donnees d'alerte sont sensibles: aucune suppression automatique.
    # Une procedure de sauvegarde/retention explicite est requise.
    pass
