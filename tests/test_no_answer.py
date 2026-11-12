from smartcattle_ai.discovery import no_answer


def silent(host, port, timeout_s):
    return False


def answers(host, port, timeout_s):
    return True


def test_a_silent_camera_says_where_it_looked_and_where_this_pc_is():
    problem = no_answer("rtsp://192.168.1.30", probe=silent, own_address="192.168.100.62")
    assert problem == "No answer from 192.168.1.30:554 (this PC is 192.168.100.62)"


def test_the_password_never_reaches_the_message():
    problem = no_answer("rtsp://admin:secreto@192.168.1.30:8554/live", probe=silent, own_address="192.168.100.62")
    assert "secreto" not in problem and "admin" not in problem and "192.168.1.30:8554" in problem


def test_a_camera_that_answers_has_no_problem():
    assert no_answer("http://192.168.1.20:8080/video", probe=answers) is None


def test_a_webcam_number_is_not_on_the_network():
    assert no_answer("1", probe=silent) is None
