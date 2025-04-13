import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os
import sys
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.utils import PERFORMANCE_MODE, logger, directionNumbers

# --- DQN Hyperparameter --- #
LEARNING_RATE = 0.001         # Lernrate für den Optimizer
GAMMA = 0.95                  # Diskontierungsfaktor für zukünftige Belohnungen
MEMORY_SIZE = 10000           # Größe des Erfahrungspuffers
BATCH_SIZE = 32               # Kleinere Batch-Größe für schnelleres Training
UPDATE_TARGET_EVERY = 500     # Frequenz für Target-Netzwerk-Updates
EPSILON_START = 1.0           # Anfangswert für Epsilon-Greedy
EPSILON_MIN = 0.05            # Minimaler Epsilon-Wert
EPSILON_DECAY = 0.995         # Abnahmerate von Epsilon
ACTION_DELAY = 60             # Ampelschaltintervall (in Ticks)
REPLAY_MIN_SIZE = 1000        # Mindestgröße für Replay-Start
UPDATE_FREQUENCY = 4          # Nur jede 4. Aktion trainieren

# --- Zustandsrepräsentation --- #
# Reduzierte Zustandsdarstellung mit nur 8 Werten:
# - 4 Werte für Fahrzeugzählung pro Richtung (Nord, Süd, Ost, West)
# - 4 Werte für Emissionsgewichtung pro Richtung
STATE_SIZE = 8

class TrafficDQNAgent:
    """Ein vereinfachter DQN-Agent für die Ampelsteuerung mit nur 2 Aktionen:
    0: Nord-Süd Grün / Ost-West Rot
    1: Nord-Süd Rot / Ost-West Grün
    """
    
    def __init__(self, action_delay=ACTION_DELAY):
        # Grundkonfiguration
        self.state_size = STATE_SIZE
        self.action_size = 2  # Nur zwei mögliche Aktionen: NS oder OW grün
        self.action_delay = action_delay  # Wie lange eine Aktion gilt
        self.tick_counter = 0  # Zählt Ticks seit der letzten Aktion
        self.update_counter = 0  # Zählt Aktualisierungen für selteneres Training
        
        # DQN-spezifische Attribute
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START  # Erkundungsrate
        
        # Modellinitialisierung
        self.model = self._build_model()          # Hauptnetzwerk für Aktionsauswahl
        self.target_model = self._build_model()   # Target-Netzwerk für stabileres Training
        self.update_target_model()                # Synchronisieren
        
        # Trainings-Effizienz: Vorhersagen in Batches
        self.prediction_states = []
        self.prediction_results = None
        self.prediction_index = 0
        self.max_prediction_batch = 32  # Größe des Batches für Vorhersagen
        
        # Zustandsverfolgung
        self.current_epoch = 0
        self.total_steps = 0
        self.epoch_steps = 0
        self.epoch_accumulated_reward = 0
        self.last_state = None
        self.last_action = None
        self.current_action_duration = 0
        
        # Metriken für Zusammenfassungen
        self.total_crashes_epoch = 0
        self.total_emissions_epoch = 0
        self.total_crossed_epoch = 0
        self.sum_avg_speeds_epoch = 0
        self.vehicle_updates_epoch = 0
        
        # Renderingkonfiguration
        self.show_render = False
        self.waiting_for_space = False
        self.render_interval = 10
        self.steps_per_epoch = 1000  # Definiert, wie viele Aktionen eine Epoche bilden
        self.total_epochs = 1000
        
        logger.info("DQN Traffic Control Agent initialisiert")
    
    def _build_model(self):
        """Erstellt ein sehr schlankes neuronales Netzwerk für DQN."""
        model = Sequential()
        
        # Eingabeschicht
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        model.add(BatchNormalization())
        
        # Versteckte Schichten - reduzierte Größe für Geschwindigkeit
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        
        # Ausgabeschicht: Q-Werte für jede Aktion
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        
        # Kompilieren mit MSE-Verlustfunktion für Q-Learning
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model
    
    def update_target_model(self):
        """Aktualisiert das Target-Netzwerk mit den Gewichten des Hauptnetzwerks."""
        self.target_model.set_weights(self.model.get_weights())
        if not PERFORMANCE_MODE:
            logger.info("Target-Netzwerk aktualisiert")
    
    def get_state(self, vehicles):
        """Generiert einen vereinfachten Zustandsvektor aus den aktuellen Fahrzeugen.
        Dieser besteht aus:
        - Anzahl der Fahrzeuge pro Richtung (Nord, Süd, Ost, West)
        - Emissionsgewichtete Anzahl pro Richtung
        """
        # Optimierte Implementierung: Gruppieren nach Richtungen
        vehicle_counts = {"right": 0, "down": 0, "left": 0, "up": 0}
        emission_weights = {"right": 0.0, "down": 0.0, "left": 0.0, "up": 0.0}
        
        # Schnelles Zählen ohne komplexe Checks
        for vehicle in vehicles:
            if hasattr(vehicle, 'direction') and not getattr(vehicle, 'crashed', False):
                direction = vehicle.direction
                if direction in vehicle_counts:
                    vehicle_counts[direction] += 1
                    
                    # Emissionsgewichtung aus vehcileClass
                    if hasattr(vehicle, 'vehicleClass'):
                        if vehicle.vehicleClass == 'bus':
                            emission_weights[direction] += 2.5
                        elif vehicle.vehicleClass == 'truck': 
                            emission_weights[direction] += 3.0
                        elif vehicle.vehicleClass == 'bike':
                            emission_weights[direction] += 0.3
                        else:  # car
                            emission_weights[direction] += 1.0
                    else:
                        emission_weights[direction] += 1.0
        
        # Umwandlung in flachen Vektor
        state = np.array([
            vehicle_counts["up"], vehicle_counts["down"],  # Nord/Süd
            vehicle_counts["right"], vehicle_counts["left"],  # Ost/West
            emission_weights["up"], emission_weights["down"],  # Nord/Süd Emission
            emission_weights["right"], emission_weights["left"]  # Ost/West Emission
        ], dtype=np.float32)
        
        # Normalisierung
        state[0:4] = state[0:4] / 50.0
        state[4:8] = state[4:8] / 100.0
        
        return state
    
    def remember(self, state, action, reward, next_state, done):
        """Speichert eine Erfahrung im Replay-Buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Wählt eine Aktion basierend auf dem aktuellen Zustand aus."""
        # Epsilon-greedy Exploration
        if np.random.rand() <= self.epsilon:
            # Zufällige Aktion
            return random.randrange(self.action_size)
        
        # Nutze die Batch-Vorhersage für bessere Leistung
        if len(self.prediction_states) == 0 or self.prediction_index >= len(self.prediction_states):
            # Wenn wir keine gespeicherten Vorhersagen haben oder alle verwendet wurden
            self.prediction_states = []
            self.prediction_index = 0
            
            # Füge den aktuellen Zustand hinzu
            self.prediction_states.append(state)
            
            # Mache die Batch-Vorhersage
            states_tensor = tf.convert_to_tensor(
                np.array(self.prediction_states), dtype=tf.float32
            )
            self.prediction_results = self.model.predict(states_tensor, verbose=0)
        
        # Nutze die aktuelle Vorhersage
        q_values = self.prediction_results[self.prediction_index]
        self.prediction_index += 1
        
        return np.argmax(q_values)
    
    def replay(self):
        """Trainiert das Netzwerk mit einer Stichprobe aus dem Erfahrungsspeicher."""
        if len(self.memory) < REPLAY_MIN_SIZE:
            return
        
        # Zufällige Stichprobe aus dem Speicher ziehen
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        # Arrays für das Batch-Training vorbereiten
        states = np.zeros((BATCH_SIZE, self.state_size))
        targets = np.zeros((BATCH_SIZE, self.action_size))
        
        # Berechne Q-Werte für alle nächsten Zustände auf einmal (Batch-Operation)
        next_states = np.array([m[3] for m in minibatch])
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Aktuelle Q-Werte für Zustände holen
        state_batch = np.array([m[0] for m in minibatch])
        current_q_values = self.model.predict(state_batch, verbose=0)
        
        # Zielwerte für jedes Element im Batch setzen
        for i, (state, action, reward, _, done) in enumerate(minibatch):
            states[i] = state
            targets[i] = current_q_values[i]  # Kopiere aktuelle Werte
            
            # Aktualisiere nur den Q-Wert für die ausgewählte Aktion
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + GAMMA * np.max(next_q_values[i])
        
        # Batch-Training
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=BATCH_SIZE)
        
        # Epsilon für die Exploration reduzieren
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def calculate_reward(self, avg_speed, crashes_this_step, emissions_this_step, newly_crossed_count):
        """Berechnet die Belohnung basierend auf CO2-Emissionen und Durchsatz.
        Überarbeitete Funktion, die intelligente Entscheidungen belohnt, nicht bloßen Durchsatz.
        """
        # Grundwert der Belohnung (klein und negativ als Anreiz, überhaupt etwas zu tun)
        base_reward = -1.0
        
        # --- Analysiere Verkehrsfluss-Effizienz ---
        
        # 1. Unfallstrafe (stark negativ)
        crash_penalty = -100.0 * crashes_this_step
        
        # 2. Verkehrsfluss: Geschwindigkeit ist wichtiger als das bloße Überqueren
        # Wenn Fahrzeuge schnell fahren, war die Ampelschaltung effizient
        flow_reward = 0.0
        if avg_speed > 2.0:  # Belohne nur, wenn die Durchschnittsgeschwindigkeit gut ist
            # Exponentiell steigende Belohnung für höhere Geschwindigkeiten
            flow_reward = 5.0 * (avg_speed ** 1.5) 
        
        # 3. Durchsatz ist nur gut, wenn die Geschwindigkeit auch stimmt
        # Ein hoher Durchsatz bei niedriger Geschwindigkeit bedeutet Stau
        throughput_reward = 0.0
        if newly_crossed_count > 0:
            if avg_speed > 2.5:  # Guter Fluss
                throughput_reward = 5.0 * newly_crossed_count
            elif avg_speed > 1.5:  # Mäßiger Fluss
                throughput_reward = 2.0 * newly_crossed_count
            else:  # Schlechter Fluss - kaum Belohnung
                throughput_reward = 0.5 * newly_crossed_count
                
        # 4. Emissionseffizienz: Emissionen pro überquerte Fahrzeuge
        emission_penalty = -0.2 * emissions_this_step  # Grundlegende Emissionsstrafe
        
        if newly_crossed_count > 0:  # Verhindere Division durch Null
            # Berechne Emissionseffizienz - wie viel CO2 pro überquertes Fahrzeug
            emission_per_vehicle = emissions_this_step / newly_crossed_count
            
            # Belohne niedrige Emissionen pro Fahrzeug (gute Effizienz)
            if emission_per_vehicle < 1.0:  # Sehr effizient
                emission_penalty = 0  # Keine Strafe
            elif emission_per_vehicle < 3.0:  # Mäßig effizient
                emission_penalty = -0.1 * emissions_this_step
                
        # Gesamtbelohnung berechnen
        reward = base_reward + flow_reward + throughput_reward + emission_penalty + crash_penalty
        
        # Belohnung in einem vernünftigen Rahmen halten (-100 bis +100)
        return np.clip(reward, -100.0, 100.0)
    
    def get_current_vehicle_count(self):
        """Bestimmt die Anzahl aktiver Fahrzeuge in der Simulation."""
        from shared.utils import simulation
        active_vehicles = [v for v in simulation if hasattr(v, 'crashed') and not v.crashed]
        return active_vehicles
    
    def end_epoch(self):
        """Behandelt Logik am Ende einer Epoche."""
        if not PERFORMANCE_MODE:
            print(f"\n--- Ende von Epoche {self.current_epoch}. Verarbeitung... ---")
            
        avg_speed_epoch = self.sum_avg_speeds_epoch / max(1, self.vehicle_updates_epoch)
        
        # Formatierung der Zusammenfassungsnachricht
        if not PERFORMANCE_MODE:
            print("=" * 60)
            print(f"EPOCHE {self.current_epoch}/{self.total_epochs} ZUSAMMENFASSUNG")
            print("-" * 60)
            print(f"Gesamtbelohnung: {self.epoch_accumulated_reward:.2f}")
            print(f"Durchschn. Geschwindigkeit: {avg_speed_epoch:.2f} Einheiten/Schritt")
            print(f"Unfälle: {self.total_crashes_epoch}")
            print(f"CO2-Emissionen: {self.total_emissions_epoch:.2f} Einheiten")
            print(f"Überquert: {self.total_crossed_epoch} Fahrzeuge")
            print(f"Epsilon: {self.epsilon:.4f}")
                
            if (self.current_epoch + 1) % self.render_interval == 0 and self.current_epoch >= 0:
                print("-" * 60)
                print("VISUALISIERUNG WIRD für die nächste Epoche AKTIVIERT - Drücke LEERTASTE zum Fortsetzen")
                
            print("=" * 60 + "\n")
        else:
            epoch_num_str = f"Epoch {self.current_epoch}/{self.total_epochs}"
            reward_str = f"Reward: {self.epoch_accumulated_reward:<10.2f}"
            crashes_str = f"Crashes: {self.total_crashes_epoch:<4}"
            crosses_str = f"Crossed: {self.total_crossed_epoch:<4}"
            avg_speed_str = f"AvgSpeed: {avg_speed_epoch:<5.2f}"
            print(
                f"| {epoch_num_str:<15} | {reward_str} | {crashes_str} | {crosses_str} | {avg_speed_str} | Eps={self.epsilon:.4f} |",
                flush=True,
            )
            
        self.current_epoch += 1
        
        # Metriken für die nächste Epoche zurücksetzen
        self.epoch_accumulated_reward = 0
        self.epoch_steps = 0
        self.total_crashes_epoch = 0
        self.total_emissions_epoch = 0
        self.sum_avg_speeds_epoch = 0
        self.total_crossed_epoch = 0
        self.vehicle_updates_epoch = 0
        
        # Bestimmen, ob in der nächsten Epoche gerendert werden soll
        if self.current_epoch % self.render_interval == 0 and self.current_epoch > 0:
            self.show_render = True
            self.waiting_for_space = True
        elif self.total_epochs == 1 and self.current_epoch == 1:
            self.show_render = True
            self.waiting_for_space = True
        else:
            self.show_render = False
            self.waiting_for_space = False
    
    def should_render(self):
        """Bestimmt, ob die Simulation gerendert werden soll."""
        return self.show_render
    
    def check_for_space(self):
        """Prüft, ob die Leertaste gedrückt wurde, um das Training fortzusetzen."""
        if not self.waiting_for_space or not self.show_render:
            return False
            
        if self.show_render and pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if not PERFORMANCE_MODE:
                        print("Training wird fortgesetzt...")
                    self.waiting_for_space = False
                    self.show_render = False
                    pygame.event.clear()
                    return True
        return False
    
    def convert_action_to_lights(self, action):
        """Konvertiert eine Aktionsnummer in eine Liste aktiver Ampeln."""
        if action == 0:  # Nord-Süd grün
            return [False, True, False, True]  # [Ost, Nord, West, Süd]
        else:  # Ost-West grün
            return [True, False, True, False]  # [Ost, Nord, West, Süd]
    
    def update(
        self,
        current_vehicles,
        avg_speed,
        crashes_this_step,
        sum_sq_waiting_time,
        emissions_this_step,
        newly_crossed_count,
    ):
        """Verarbeitet einen Schritt der Simulation und entscheidet über Ampelzustände.
        Mit Optimierungen für bessere Leistung.
        """
        # Aktuellen Zustand erhalten
        current_state = self.get_state(current_vehicles)
        
        # 1. Belohnung für den letzten Zustand berechnen, wenn vorhanden
        if self.last_state is not None and self.last_action is not None:
            reward = self.calculate_reward(
                avg_speed,
                crashes_this_step,
                emissions_this_step,
                newly_crossed_count,
            )
            self.epoch_accumulated_reward += reward
            
            # Metriken für Epochenzusammenfassung aktualisieren
            self.total_crashes_epoch += crashes_this_step
            self.total_emissions_epoch += emissions_this_step
            self.total_crossed_epoch += newly_crossed_count
            self.sum_avg_speeds_epoch += avg_speed
            self.vehicle_updates_epoch += 1
            
            # 2. Erfahrung nur speichern, wenn die ACTION_DELAY-Periode vorbei ist
            if self.tick_counter >= self.action_delay:
                done = False  
                self.remember(self.last_state, self.last_action, reward, current_state, done)
                
                # Netzwerk trainieren, aber nur jede N-te Aktion für bessere Leistung
                self.update_counter += 1
                if self.update_counter >= UPDATE_FREQUENCY:
                    self.replay()
                    self.update_counter = 0
                
                # Target-Netzwerk aktualisieren
                if self.total_steps % UPDATE_TARGET_EVERY == 0:
                    self.update_target_model()
                
                self.total_steps += 1
                self.epoch_steps += 1
                
                # Neue Aktion wählen
                self.last_action = self.act(current_state)
                self.tick_counter = 0
            else:
                # Während der Verzögerung keine neue Aktion wählen
                self.tick_counter += 1
        else:
            # Erste Aktion überhaupt wählen
            self.last_action = self.act(current_state)
            self.tick_counter = 0
        
        # Aktuellen Zustand für den nächsten Schritt speichern
        self.last_state = current_state
        
        # Aktionsperiode prüfen
        if self.epoch_steps >= self.steps_per_epoch:
            self.end_epoch()
        
        # 3. Aktion in Ampelbelegung umwandeln und zurückgeben
        return self.convert_action_to_lights(self.last_action)
    
    def save_model(self, filepath_prefix):
        """Speichert die Modellgewichte."""
        if self.model:
            try:
                model_path = f"{filepath_prefix}.weights.h5"
                self.model.save_weights(model_path)
                logger.info(f"DQN-Modell gespeichert unter: {model_path}")
            except Exception as e:
                logger.error(f"Fehler beim Speichern des DQN-Modells unter {filepath_prefix}: {e}")
        else:
            logger.warning("Versuch, das Modell zu speichern, aber es ist None.")
    
    def load_models(self, filepath_prefix):
        """Lädt die Modellgewichte."""
        model_path = f"{filepath_prefix}.weights.h5"
        print(f"Versuche, DQN-Modell zu laden von: {model_path}")
        
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                self.update_target_model()  # Synchronisiere mit target model
                print("DQN-Modell erfolgreich geladen.")
            except Exception as e:
                print(f"Fehler beim Laden der DQN-Modellgewichte: {e}")