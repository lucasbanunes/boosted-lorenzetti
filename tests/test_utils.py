from boosted_lorenzetti.utils import unflatten_dict


def test_unflatten_dict():
    flat_dict = {
        'a.b.c': 1,
        'a.b.d': 2,
        'a.e': 3,
    }
    expected = {
        'a': {
            'b': {
                'c': 1,
                'd': 2
            },
            'e': 3
        },
    }
    unflattened = unflatten_dict(flat_dict)
    assert unflattened == expected
