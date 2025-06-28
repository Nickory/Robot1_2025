from typing import Optional
import k210_models
import music
import basic

# Initialize global variables
face: Optional[int] = -2  # Default face ID, indicating no face detected


def initialize() -> None:
    """
    Initialize the K210 models and clear the LED screen.
    """
    k210_models.initialization()
    basic.clear_screen()


def play_music_for_face(face_id: int) -> None:
    """
    Play a specific melody based on the detected face ID and clear the screen.

    Args:
        face_id (int): The ID of the detected face (0, 1, or 2).
    """
    melodies = {
        0: music.built_in_playable_melody(Melodies.POWER_UP),
        1: music.built_in_playable_melody(Melodies.JUMP_UP),
        2: music.built_in_playable_melody(Melodies.JUMP_DOWN)
    }

    melody = melodies.get(face_id)
    if melody:
        music._play_default_background(melody, music.PlaybackMode.UNTIL_DONE)
        basic.clear_screen()


def monitor_face_detection() -> None:
    """
    Continuously monitor face detection and display the face ID on the LED screen.
    """
    global face
    face = k210_models.face_reg()
    if face >= 0:
        basic.show_number(face)


def main() -> None:
    """
    Main function to set up and run the face detection and music playback loops.
    """
    initialize()

    # Set up concurrent forever loops for music playback and face detection
    basic.forever(play_music_for_face, face)
    basic.forever(monitor_face_detection)


if __name__ == "__main__":
    main()