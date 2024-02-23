# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
from irlc.ex00.fruit_homework import add, misterfy, mean_value, fruits_ordered, BasicFruitShop, OnlineFruitShop, shop_smart
from unitgrade import hide

class AdditionQuestion(UTestCase):
    """ Problem 1: Adding two numbers """
    def test_add(self):
        """ Adding two numbers together """
        self.assertEqual(add(2, 3), 5) # Test the add-function.
        self.assertEqual(add(2, -917), -915) # Test the add-function.



class MisterfyQuestion(UTestCase):
    """ Problem 2: Misterfy a list """
    def test_misterfy(self):
        """ Add 'mr' in front of each item in a string """
        self.assertEqualC(misterfy(['dog', 'cat', 'lion']))
        self.assertEqualC(misterfy(['giraffe']))
        self.assertEqualC(misterfy([]))



class MeanOfDie(UTestCase):
    """ Problem 3: Mean of die """
    def test_mean_value(self):
        """ Compute mean of two dice """
        p_die = {1: 0.20,
                 2: 0.10,
                 3: 0.15,
                 4: 0.05,
                 5: 0.10,
                 6: 0.40}
        self.assertL2(mean_value(p_die), tol=0.0001)
        self.assertL2(mean_value({-1: 0.5, 1: 0.5}), tol=0.0001)



class FruitsOrdered(UTestCase):
    """ Problem 4: The fruits_ordered function """
    def test_fruits_ordered(self):
        """ fruits_ordered """
        order = {'apples': 1.0,
                 'oranges': 3.0}
        self.assertEqualC(list(sorted(fruits_ordered(order))))
        order2 = {'banana': 4,
                  'apples': 1.0,
                  'oranges': 3.0,
                  'pears': 4}
        self.assertEqualC(list(sorted(fruits_ordered(order2))))


class BasicClass(UTestCase):
    """ Problem 5: The BasicFruitShop """
    def test_cost(self):
        """ Testing cost function """
        price1 = {"apple": 4, "pear": 8, 'orange': 10}
        shop1 = BasicFruitShop("Alis Funky Fruits", price1)
        self.assertEqualC(shop1.cost("apple"))
        self.assertEqualC(shop1.cost("pear"))

        price2 = {'banana': 9, "apple": 5, "pear": 7, 'orange': 11}
        shop2 = BasicFruitShop("Hansen Fruit Emporium", price2)
        self.assertEqualC(shop2.cost("orange"))
        self.assertEqualC(shop2.cost("banana"))


class Inheritance(UTestCase):
    title = "Problem 6: Inheritance"

    def test_price_of_order(self):
        """ Testing the price_of_order function """
        price_of_fruits = {'apples': 2, 'oranges': 1, 'pears': 1.5, 'mellon': 10, 'banana': 1.5}
        shopA = OnlineFruitShop('shopA', price_of_fruits)

        order1 = {'apples': 1.0,
                 'oranges': 3.0}
        self.assertL2(shopA.price_of_order(order1), tol=1e-8)
        order2 = {'banana': 4,
                  'apples': 1.0,
                  'oranges': 3.0,
                  'pears': 4}
        self.assertL2(shopA.price_of_order(order2), tol=1e-8)


class ClassUse(UTestCase):
    title = "Problem 7: Using classes"

    def test_shop_smarter(self):
        """ Testing the shop_smarter function """
        price_of_fruits = {'apples': 2, 'oranges': 1, 'pears': 1.5, 'mellon': 10}
        shopA = OnlineFruitShop('shopA', price_of_fruits)
        shopB = OnlineFruitShop('shopB', {'apples': 1.0, 'oranges': 5.0})

        shops = [shopA, shopB]
        order = {'apples': 1.0,
                 'oranges': 3.0}
        self.assertEqualC(shop_smart(order, shops).name)
        order = {'apples': 3.0}  # test with a new order.
        self.assertEqualC(shop_smart(order, shops).name)


class FruitReport(Report):
    title = "Fruit example report"
    abbreviate_questions = True
    questions = [(AdditionQuestion, 10),
                 (MisterfyQuestion, 10),
                 (MeanOfDie, 10),
                 (FruitsOrdered, 10),
                 (BasicClass, 10),
                 (Inheritance, 10),
                 (ClassUse, 10)]

    import irlc
    pack_imports = [irlc]


if __name__ == "__main__":
    from unitgrade import evaluate_report_student
    evaluate_report_student(FruitReport())
