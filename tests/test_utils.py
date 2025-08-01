import json
from boosted_lorenzetti.utils import unflatten_dict
from typing import List
from pydantic import BaseModel


def test_unflatten_dict():
    flat_dict = {
        'a.b.c': 1,
        'a.b.d': 2,
        'a.e': 3,
        'b': [
            1,
            2,
            {
                'x': 10,
                'y': 20,
                'z.p': 30,
                'z.q': 40
            },

        ]
    }
    expected = {
        'a': {
            'b': {
                'c': 1,
                'd': 2
            },
            'e': 3
        },
        'b': [
            1,
            2,
            {
                'x': 10,
                'y': 20,
                'z': {
                    'p': 30,
                    'q': 40
                }
            }
        ]
    }
    unflattened = unflatten_dict(flat_dict)
    assert unflattened == expected
