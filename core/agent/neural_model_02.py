import numpy as np
import pygame
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.utils import PERFORMANCE_MODE
import logging

logger = logging.getLogger(__name__)

# --- Einfache Hyperparameter --- #
LEARNING_RATE = 1e-3  # Höhere Lernrate für schnellere Konvergenz
BUFFER_SIZE = 10000   # Größe des Erfahrungspuffers
BATCH_SIZE = 128      # Größe der Trainingsbatches
UPDATE_FREQUENCY = 128 # Wie oft das Modell trainiert wird
EXPLORATION_RATE = 0.2 # Erkundungsrate für zufällige Aktionen
TARGET_REWARD = 1_000_000.0 # Optimales Zielbild - sehr hoher Reward als "perfekt"
MAX_VEHICLES = 50     # Maximale Anzahl von Fahrzeugen im Zustand
VEHICLE_FEATURES = 4  # Fahrzeugmerkmale: Distanz, Geschwindigkeit, Wartezeit, Winkel
INPUT_SIZE = MAX_VEHICLES * VEHICLE_FEATURES # Größe des Eingabevektors
INTERSECTION_CENTER_X = 700  # Zentrum der Kreuzung (X-Koordinate)
INTERSECTION_CENTER_Y = 400  # Zentrum der Kreuzung (Y-Koordinate)
# --- Ende Hyperparameter --- #

class SimpleTrafficController:
    """Ein einfacherer Verkehrsregler mit einem einzelnen neuronalen Netz."""
    
    def __init__(self, steps_per_epoch=10000, total_epochs=50):
        # Konfiguration
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.input_size = INPUT_SIZE
        self.output_size = 4  # 4 Ampelphasen (eine für jede Richtung)
        self.render_interval = 10
        
        # Zustandsverfolgung
        self.experience_buffer = deque(maxlen=BUFFER_SIZE)
        self.total_steps = 0
        self.current_epoch = 0
        self.epoch_accumulated_reward = 0
        self.epoch_steps = 0
        self.show_render = False
        self.waiting_for_space = False
        
        # Metrikerfassung für Epochenzusammenfassung
        self.total_crashes_epoch = 0
        self.total_emissions_epoch = 0
        self.sum_avg_speeds_epoch = 0
        self.vehicle_updates_epoch = 0
        
        # Letzter Zustand für Training
        self.last_state = np.zeros(self.input_size)
        self.last_action = None
        
        # Modell initialisieren
        try:
            if not PERFORMANCE_MODE:
                print("Initialisiere einfaches neuronales Netz...")
            self.model = self._build_model()
            if not PERFORMANCE_MODE:
                print("Neuronales Netz erfolgreich initialisiert")
                self.model.summary()
        except Exception as e:
            print(f"FEHLER beim Initialisieren des Modells: {e}")
            print("Nutze zufällige Aktionen als Fallback")
            self.model = None
    
    def _build_model(self):
        """Erstellt ein einfaches neuronales Netz mit einer einzelnen Ausgabeschicht."""
        model = Sequential()
        # Eingabeschicht mit Normalisierung
        model.add(Input(shape=(self.input_size,)))
        model.add(BatchNormalization())
        
        # Versteckte Schichten
        model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        
        # Ausgabeschicht - 4 Ausgaben für die 4 Ampelrichtungen mit Sigmoid-Aktivierung
        model.add(Dense(self.output_size, activation='sigmoid'))
        
        # MSE als Verlustfunktion (versucht, den Reward zu maximieren)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mse'
        )
        return model
    
    def get_state(self, vehicles):
        """Wandelt Fahrzeugdaten in einen flachen Zustandsvektor um.
        
        Args:
            vehicles: Pygame.sprite.Group mit Vehicle-Objekten
            
        Returns:
            Ein flaches NumPy-Array, das den Zustand darstellt.
        """
        state_features = []
        center_x = INTERSECTION_CENTER_X
        center_y = INTERSECTION_CENTER_Y
        
        if not vehicles:  # Leere Gruppe behandeln
            return np.zeros(self.input_size)
            
        # Zonen für die 4 Richtungen
        zones = {
            "right": [],  # Fahrzeuge von rechts kommend
            "down": [],   # Fahrzeuge von oben kommend
            "left": [],   # Fahrzeuge von links kommend
            "up": []      # Fahrzeuge von unten kommend
        }
        
        # Fahrzeuge nach Zonen gruppieren
        for vehicle in vehicles:
            if vehicle.crashed:  # Beschädigte Fahrzeuge überspringen
                continue
                
            try:
                # Fahrzeugeigenschaften extrahieren
                dx = vehicle.rect.centerx - center_x
                dy = vehicle.rect.centery - center_y
                distance = math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)  # Radians von -pi bis pi
                speed = min(vehicle.speed, 5.0) / 5.0  # Normalisiert auf [0,1]
                wait_time = min(vehicle.waiting_time, 50.0) / 50.0  # Normalisiert auf [0,1]
                
                # Fahrzeug der richtigen Zone zuordnen
                if hasattr(vehicle, 'direction'):
                    direction = vehicle.direction
                    if direction in zones:
                        zones[direction].append([
                            distance, 
                            speed,
                            wait_time,
                            angle
                        ])
            except AttributeError as e:
                logger.error(f"Fehler beim Zugriff auf Fahrzeugattribute: {e}. Fahrzeug: {vehicle}")
                continue
                
        # Jede Zone verarbeiten und Merkmale hinzufügen
        vehicles_per_zone = MAX_VEHICLES // 4
        for direction in zones:
            # Nach Distanz sortieren (nächste zuerst)
            zones[direction].sort(key=lambda x: x[0])
            
            # Die nächsten N Fahrzeuge der Zone verwenden
            for i in range(min(len(zones[direction]), vehicles_per_zone)):
                state_features.extend(zones[direction][i])
                
            # Padding für diese Zone
            padding_needed = vehicles_per_zone - min(len(zones[direction]), vehicles_per_zone)
            if padding_needed > 0:
                state_features.extend([0.0] * (padding_needed * VEHICLE_FEATURES))
                
        # Konsistenzprüfung der Größe
        if len(state_features) != self.input_size:
            if len(state_features) < self.input_size:
                state_features.extend([0.0] * (self.input_size - len(state_features)))
            else:
                state_features = state_features[:self.input_size]
                
        return np.array(state_features, dtype=np.float32)
    
    def get_action(self, state):
        """Bestimmt die Aktion (Ampelzustände) für den aktuellen Zustand.
        
        Args:
            state: Ein NumPy-Array mit dem aktuellen Zustand
            
        Returns:
            Eine Liste mit Boolean-Werten für die Ampelzustände
        """
        if self.model is None or random.random() < EXPLORATION_RATE:
            # Zufällige Aktion: 1-2 zufällige Ampeln auf grün
            num_active = random.randint(1, 2)
            action_indices = random.sample(range(self.output_size), num_active)
            return [i in action_indices for i in range(self.output_size)]
        
        # Normale Aktion mit dem Modell
        # Stelle sicher, dass state die richtige Form hat
        if state.shape != (self.input_size,):
            try:
                state = np.reshape(state, (self.input_size,))
            except ValueError:
                state = np.zeros(self.input_size)
        
        # Modell für die Vorhersage verwenden
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.model(state_tensor, training=False)[0]
        
        # Schwellenwert auf Wahrscheinlichkeiten anwenden (>0.5 = grün)
        return [prob > 0.5 for prob in action_probs]
    
    def calculate_reward(
        self,
        avg_speed,
        crashes_this_step,
        sum_sq_waiting_time,
        emissions_this_step,
        newly_crossed_count
    ):
        """Berechnet die Belohnung für den aktuellen Simulationsschritt.
        
        Vereinfachte Belohnungsfunktion mit dem Ziel, die Belohnung auf TARGET_REWARD zu maximieren.
        
        Args:
            avg_speed: Durchschnittliche Geschwindigkeit der Fahrzeuge
            crashes_this_step: Anzahl der Kollisionen in diesem Schritt
            sum_sq_waiting_time: Summe der quadratischen Wartezeiten
            emissions_this_step: Emissionen in diesem Schritt
            newly_crossed_count: Anzahl der Fahrzeuge, die die Kreuzung überquert haben
            
        Returns:
            Die berechnete Belohnung
        """
        # Anzahl der aktiven Fahrzeuge bestimmen
        vehicle_count = len(self.get_current_vehicle_count())
        
        # --- Einfache Gewichtungen ---
        throughput_weight = 10.0         # Hohe Belohnung für Durchsatz (Geschwindigkeit × Fahrzeuganzahl)
        crossing_reward = 25.0           # Belohnung pro Fahrzeug, das die Kreuzung überquert
        waiting_penalty = -0.01          # Mildere Strafe für Wartezeiten
        emission_penalty = -0.1          # Reduzierte Strafe für Emissionen
        crash_penalty = -50.0            # Strafe für Kollisionen
        
        # Normalisierung
        normalized_speed = min(avg_speed / 3.0, 1.0)  # Max speed normalisiert auf 1.0
        
        # --- Belohnungsberechnung ---
        reward = 0
        
        # Hauptbelohnung: Durchsatz (speed × count) und Überquerungen
        throughput = normalized_speed * vehicle_count
        reward += throughput_weight * throughput
        reward += crossing_reward * newly_crossed_count
        
        # Strafen
        reward += waiting_penalty * sum_sq_waiting_time
        reward += emission_penalty * emissions_this_step
        reward += crash_penalty * crashes_this_step
        
        # Rescaling: Belohnungen in die Nähe des Zielwerts bringen (aber nie überschreiten)
        if reward > 0:
            # Positive Belohnungen in Richtung TARGET_REWARD skalieren
            reward = min(reward, TARGET_REWARD * 0.8)  # Max 80% des Ziels erreichen
        
        return reward
    
    def get_current_vehicle_count(self):
        """Bestimmt die Anzahl aktiver Fahrzeuge in der Simulation."""
        from shared.utils import simulation
        active_vehicles = [v for v in simulation if hasattr(v, 'crashed') and not v.crashed]
        return active_vehicles
    
    def store_experience(self, state, action, reward, next_state):
        """Speichert eine Erfahrung im Puffer."""
        # Wandle action in ein binäres Array um
        action_array = np.array([1.0 if a else 0.0 for a in action], dtype=np.float32)
        self.experience_buffer.append((state, action_array, reward, next_state))
    
    def train_model(self):
        """Trainiert das Modell mit den gespeicherten Erfahrungen."""
        if len(self.experience_buffer) < BATCH_SIZE:
            logger.warning(f"Nicht genug Erfahrungen für Training: {len(self.experience_buffer)}/{BATCH_SIZE}")
            return
            
        # Zufällige Stichprobe aus dem Puffer ziehen
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            min(BATCH_SIZE, len(self.experience_buffer)), 
            replace=False
        )
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Daten aus dem Batch extrahieren
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        
        # Target-Werte basierend auf rewards erstellen:
        # Macht den Reward zum "Ziel" für das Netzwerk
        target_values = np.zeros((len(batch), self.output_size), dtype=np.float32)
        
        for i, (_, action, reward, _) in enumerate(batch):
            # Scale: Reward in [0,1] für Sigmoid-Aktivierung
            scaled_reward = min(max(reward / TARGET_REWARD, 0.0), 1.0)
            
            # Für aktive Ampeln: Maximiere es auf 1.0 wenn Belohnung positiv
            # Für inaktive Ampeln: Minimiere es auf 0.0 wenn Belohnung positiv
            target_values[i] = action * scaled_reward + (1 - action) * (1 - scaled_reward)
        
        # Modell trainieren
        history = self.model.fit(
            states, 
            target_values,
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=0
        )
        
        if not PERFORMANCE_MODE:
            loss = history.history['loss'][0] if history.history['loss'] else float('nan')
            print(f"Training: Loss={loss:.6f}, Buffer={len(self.experience_buffer)}")
    
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
            
            if (self.current_epoch + 1) % self.render_interval == 0 and self.current_epoch >= 0:
                print("-" * 60)
                print("VISUALISIERUNG WIRD für die nächste Epoche AKTIVIERT - Drücke LEERTASTE zum Fortsetzen")
                
            print("=" * 60 + "\n")
        else:
            epoch_num_str = f"Epoch {self.current_epoch}/{self.total_epochs}"
            reward_str = f"Reward: {self.epoch_accumulated_reward:<10.2f}"
            crashes_str = f"Crashes: {self.total_crashes_epoch:<4}"
            avg_speed_str = f"AvgSpeed: {avg_speed_epoch:<5.2f}"
            print(
                f"| {epoch_num_str:<15} | {reward_str} | {crashes_str} | {avg_speed_str} |",
                flush=True,
            )
            
        self.current_epoch += 1
        
        # Metriken für die nächste Epoche zurücksetzen
        self.epoch_accumulated_reward = 0
        self.epoch_steps = 0
        self.total_crashes_epoch = 0
        self.total_emissions_epoch = 0
        self.sum_avg_speeds_epoch = 0
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
    
    def update(
        self,
        current_vehicles,
        avg_speed,
        crashes_this_step,
        sum_sq_waiting_time,
        emissions_this_step,
        newly_crossed_count,
    ):
        """Verarbeitet einen Schritt der Simulation: berechnet Belohnung, speichert Erfahrung, entscheidet Aktion.
        
        Args:
            current_vehicles: Pygame.sprite.Group mit aktuellen Fahrzeugobjekten.
            avg_speed: Durchschnittliche Geschwindigkeit der Fahrzeuge im vorherigen Schritt.
            crashes_this_step: Anzahl der Kollisionen im vorherigen Schritt.
            sum_sq_waiting_time: Summe der quadrischen Wartezeiten der Fahrzeuge.
            emissions_this_step: Gesamtemissionen der Fahrzeuge im vorherigen Schritt.
            newly_crossed_count: Anzahl der Fahrzeuge, die die Kreuzung überquert haben.
            
        Returns:
            Eine Liste von Boolean-Werten für die Ampelzustände im nächsten Schritt.
        """
        # 1. Zustand vom aktuellen Fahrzeugzustand erhalten
        current_state = self.get_state(current_vehicles)
        
        # 2. Belohnung basierend auf dem Ergebnis des vorherigen Schritts/Aktion berechnen
        reward = 0.0
        if self.last_state is not None:
            reward = self.calculate_reward(
                avg_speed,
                crashes_this_step,
                sum_sq_waiting_time,
                emissions_this_step,
                newly_crossed_count,
            )
            self.epoch_accumulated_reward += reward
            
            # Metriken für die Epochenzusammenfassung aktualisieren
            self.total_crashes_epoch += crashes_this_step
            self.total_emissions_epoch += emissions_this_step
            self.sum_avg_speeds_epoch += avg_speed
            self.vehicle_updates_epoch += 1
            
            # Speichern der vorherigen Erfahrung
            if self.last_action is not None:
                self.store_experience(
                    self.last_state,
                    self.last_action,
                    reward,
                    current_state
                )
        
        # 3. Aktion für den aktuellen Zustand wählen
        action = self.get_action(current_state)
        
        # 4. "Letzte" Variablen für die nächste Iteration aktualisieren
        self.last_state = current_state
        self.last_action = action
        
        # Schrittczähler aktualisieren
        self.total_steps += 1
        self.epoch_steps += 1
        
        # 5. Modelltraining auslösen, wenn genügend Schritte vergangen sind
        if self.total_steps % UPDATE_FREQUENCY == 0:
            if not PERFORMANCE_MODE:
                logger.info(f"--- Training nach {self.total_steps} Schritten ---")
            self.train_model()
        
        return action
    
    def save_model(self, filepath_prefix):
        """Speichert die Modellgewichte."""
        if self.model:
            try:
                # Dateinamen korrigieren, damit er mit .weights.h5 endet, anstatt _simple_model.h5 anzuhängen
                model_path = f"{filepath_prefix}.weights.h5"
                self.model.save_weights(model_path)
                logger.info(f"Modell gespeichert unter: {model_path}")
            except Exception as e:
                logger.error(f"Fehler beim Speichern des Modells unter {filepath_prefix}: {e}")
        else:
            logger.warning("Versuch, das Modell zu speichern, aber es ist None.")
    
    def load_models(self, filepath_prefix):
        """Lädt die Modellgewichte."""
        # Dateinamen korrigieren, damit er mit .weights.h5 endet
        model_path = f"{filepath_prefix}.weights.h5"
        print(f"Versuche, Modell zu laden von: {model_path}")
        
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                print("Modell erfolgreich geladen.")
            except Exception as e:
                print(f"Fehler beim Laden der Modellgewichte: {e}")
        else:
            print(f"Modell nicht gefunden: {model_path}")