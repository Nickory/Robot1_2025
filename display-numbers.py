from microbit import *


def show_number(num: int, duration: int = 1000) -> None:
    """
    Display a number on the LED screen for a specified duration.

    Args:
        num (int): The number to display (0-9).
        duration (int, optional): Duration to display the number in milliseconds. Defaults to 1000ms.

    Returns:
        None
    """
    if 0 <= num <= 9:
        display.show(str(num))  # Convert to string for better compatibility
        sleep(duration)
    else:
        display.scroll("ERR")  # Display error message for invalid input


# Mapping of face IDs to numbers (extensible for future face recognition integration)
face_to_number = {
    1: 1,  # Face ID 1 maps to number 1
    2: 2,  # Face ID 2 maps to number 2
    3: 3  # Face ID 3 maps to number 3
}


def mock_face_detection() -> int | None:
    """
    Simulate face detection using button inputs (for testing purposes).

    Returns:
        int | None: Simulated face ID (1 or 2) or None if no face is detected.
    """
    if button_a.was_pressed():
        return 1  # Simulate detection of face ID 1
    elif button_b.was_pressed():
        return 2  # Simulate detection of face ID 2
    return None  # No face detected


def main() -> None:
    """
    Main loop for handling face detection and number display.

    Continuously checks for simulated face detection and displays the corresponding number.
    """
    while True:
        detected_face = mock_face_detection()

        if detected_face is not None:
            # Retrieve the corresponding number from the mapping, default to 0 for unknown faces
            target_num = face_to_number.get(detected_face, 0)
            show_number(target_num)

        sleep(100)  # Reduce CPU usage with a short delay


if __name__ == "__main__":
    main()