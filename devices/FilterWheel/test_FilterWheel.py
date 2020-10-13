from unittest import TestCase
from devices.FilterWheel.DummyFilterWheel import FilterWheel

filter_wheel = FilterWheel()


class TestFilterWheel(TestCase):
    def test_2_a_position_names_dict(self):
        filter_wheel.position_names_dict = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f'}
        self.assertDictEqual(filter_wheel.position_names_dict, {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f'})

    def test_3_position(self):
        filter_wheel.position = 1
        self.assertEqual(filter_wheel.position['number'], 1)
        filter_wheel.position = 'a'
        self.assertEqual(filter_wheel.position['number'], 1)
        filter_wheel.position = 0
        self.assertEqual(filter_wheel.position['number'], 1)
        filter_wheel.position = 4
        self.assertEqual(filter_wheel.position['number'], 4)
        filter_wheel.position = 'a'
        self.assertEqual(filter_wheel.position['number'], 1)

    def test_4_id(self):
        self.assertEqual(filter_wheel.id, 'THORLABS FW102C/FW212C Filter Wheel version 1.07')

    def test_5_speed(self):
        self.assertEqual(filter_wheel.speed, 1)
        filter_wheel.speed = 'slow'
        self.assertEqual(filter_wheel.speed, 0)
        filter_wheel.speed = 'fast'
        self.assertEqual(filter_wheel.speed, 1)

    def test_6_position_count(self):
        self.assertEqual(filter_wheel.position_count, 6)

    def test_7_b_is_pos_name_val(self):
        self.assertTrue(filter_wheel.is_position_name_valid('b'))
        self.assertFalse(filter_wheel.is_position_name_valid('ab'))
