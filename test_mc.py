from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from core.rocket_state import RocketStateEngine
from core.monte_carlo_engine import MonteCarloEngine, MonteCarloConfig
import sys

app = QApplication(sys.argv)

engine = RocketStateEngine()
engine.state.dry_mass = 10.0
engine.state.length = 2.0
engine.state.diameter = 0.15

mc = MonteCarloEngine(engine)
cfg = MonteCarloConfig(num_simulations=5)

def on_finished(r):
    print("Success! Mean Apogee:", r.apogee_mean)
    app.quit()

def on_failed(e):
    print("Failed!", e)
    app.quit()

mc.analysis_finished.connect(on_finished)
mc.analysis_failed.connect(on_failed)

print("Starting MC analysis...")
mc.start(cfg)

# Fallback timeout
QTimer.singleShot(15000, lambda: (print("Timeout!"), app.quit()))

app.exec()
