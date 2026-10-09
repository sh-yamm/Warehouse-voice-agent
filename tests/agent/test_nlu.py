from voiceagent.agent.nlu import grounded, split_quantity


def test_grounded_accepts_customer_words():
    assert grounded("two packets of Amul milk", "Can you also add two packets of Amul milk?")
    assert grounded("42 Church Street, Bengaluru", "No, it's 42 Church Street, Bengaluru.")
    assert grounded("tomorrow evening", "Tomorrow evening works for me.")
    assert grounded("leave it with the security guard", "Please leave it with the security guard.")


def test_grounded_rejects_agent_words():
    assert not grounded("Amul milk and eggs", "Yes, that sounds good.")
    assert not grounded("Aashirvaad sugar", "Yes, that sounds good.")
    assert not grounded("23 MG Road, Bengaluru", "Yes, that's right.")
    assert not grounded("", "Anything at all")
    assert not grounded("the", "the")


def test_split_quantity():
    assert split_quantity("two packets of Amul milk") == (2, "Amul milk")
    assert split_quantity("3 Haldiram salted peanuts") == (3, "Haldiram salted peanuts")
    assert split_quantity("Amul milk") == (1, "Amul milk")
    assert split_quantity("an extra pack of eggs") == (1, "eggs")
