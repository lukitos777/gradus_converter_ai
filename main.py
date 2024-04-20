from typing import List
from typing import Union
from asyncio import gather
from asyncio import run
# * use only when you uncomment generator function
# * from random import uniform

class Neiron:
    def __init__(self, weight: float = 1.0, bias: float = 1.0) -> None:
        self.weight = weight
        self.bias = bias

    def convert(self, celsius: float) -> float:
        return self.weight * celsius + self.bias
    
    def train(
        self, celsius_vector: List[float],
        farengeit_vector: List[float],
        learning_rate: float
    ):
        for i in range(len(celsius_vector)):
            result: float = self.convert(celsius=celsius_vector[i])
            error: float = result - farengeit_vector[i]
            gradient_weight: float = error * celsius_vector[i]
            gradient_bias: float = error

            self.weight -= learning_rate * gradient_weight
            self.bias -= learning_rate * gradient_bias

# ! to generate data set
# * use only if you want to rise data set volume
'''
def store_data():
    with open(file='celsiusData.txt', mode='a') as Data,\
        open(file='farengeitData.txt', mode='a') as Res_DATA:
        for i in range(100):
            generated_data = uniform(100, 200)
            result_data = generated_data * 1.8 + 32
            Data.write(f'{generated_data}\n')
            Res_DATA.write(f'{result_data}\n')
'''
async def read(file_name: str) -> List[float]:
    with open(file=file_name, mode='r') as file:
        return [round(float(x), 2) for x in file.readlines()]
    
async def task_meneger() -> Union[List[str], List[str]]:
    task1 = read(file_name='celsiusData.txt')
    task2 = read(file_name='farengeitData.txt')
    return await gather(task1, task2)

def main():
    # * data is already generated
    # ? store_data()
    data1, data2 = run(task_meneger())

    neiron = Neiron()
    neiron.train(celsius_vector=data1, farengeit_vector=data2, learning_rate=0.00025)

    print(neiron.convert(25.0))
    print(neiron.weight)
    print(neiron.bias)

if __name__ == '__main__':
    main()