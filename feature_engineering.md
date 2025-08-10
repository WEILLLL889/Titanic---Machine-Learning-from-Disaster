# 初始特征


        print(train_data.dtypes)


The dataset contains the following features:

|name | dtype| meaning |values |
| ----- | ---- | -------- | -------- |
PassengerId   |   int64| Unique identifier for each passenger | 1 to 891
Survived      |   int64| Survival (0 = No; 1 = Yes) | 0, 1
Pclass        |   int64| Ticket class (1 = 1st; 2 = 2nd; 3 = 3rd) | 1, 2, 3
Name          | object | Name of the passenger | e.g., "Braund, Mr. Owen Harris 
Sex            | object| sex of the passenger |male,female
| Age           | float64| Age in years | 0.42 to 80
SibSp         |   int64| # of siblings / spouses aboard the Titanic | 0 to 8
Parch         |   int64| # of parents / children aboard the Titanic | 0 to 6
Ticket        | object | Ticket number | e.g., "A/5 21171"
Fare          | float64| Passenger fare | 0 to 512.3292
Cabin         | object | Cabin number | e.g., "C85"
Embarked      | object | Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton) | C, Q, S

# load data
        import pandas as pd
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')

# 特征工程
## 处理 Nan

        #setting silly values to nan
        df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
        
        #Special case for cabins as nan may be signal
        df.Cabin = df.Cabin.fillna('Unknown')    

## Title
- 含义 ：从姓名中提取称谓，把姓名映射成四类 Mr, Mrs, Miss ，Master 
定义一个内部函数 replace_titles，接受一行（x，通常是 pandas Series）

        def replace_titles(x):
            title=x['Title']
            if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
                return 'Mr'
            elif title in ['Countess', 'Mme']:
                return 'Mrs'
            elif title in ['Mlle', 'Ms']:
                return 'Miss'
            elif title =='Dr':
                if x['Sex']=='Male':
                    return 'Mr'
                else:
                    return 'Mrs'
            else:
                return title

 根据 x['Title'] 的值做映射/合并：
- 把 Don, Major, Capt, Jonkheer, Rev, Col 等都当作 'Mr'（男性头衔归入 Mr）。

- 把 Countess, Mme 归为 'Mrs'。

- 把 Mlle, Ms 归为 'Miss'。

- 对于 'Dr' 会看 Sex 列：如果 Sex=='Male' 返回 'Mr'，否则返回 'Mrs'（把医生按性别分为 Mr/Mrs）。

- 其他情况直接返回原 title（没有覆盖到的称谓保持不变）。

- 在 DataFrame 上逐行应用 replace_titles，并把结果覆盖回 Title 列，从而把称谓做类别合并（标准化）。
  
        df['Title']=df.apply(replace_titles, axis=1)
## Cabin - Deck
- 含义：提取船舱号的首字母作为新特征，把 Cabin（ "C123"）里的甲板/楼层信息提取出来，命名为 Deck。
  
        df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))


- 把 Cabin 列的每个值传给 substrings_in_string，从中提取出在 cabin_list 中出现的子串（例如 Cabin="C85" 会匹配 'C'），结果存到新列 Deck。

- 如果 Cabin 为 'Unknown'，substrings_in_string 应会返回 'Unknown'。



## Ticket_number
- 含义：提取 Ticket 列中的数字部分，作为新特征 Ticket
  

## Ticket_item
- 含义：提取 Ticket 列中的字母部分，作为新特征 Ticket_item。

        


## Family_Size
- 含义：家庭成员数量，计算每个乘客的家庭成员数量，即 SibSp + Parch 。

        df['Family_Size'] = df['SibSp'] + df['Parch'] 


## data type dictionary
接下来构造一个字典，用来记录各列的数据类型类别（方便后续处理或特征工程/建模时依据类型选择变换）。

        data_type_dict={'Pclass':'ordinal', 
                        'Sex':'nominal', 
                        'Age':'numeric', 
                        'Fare':'numeric', 
                        'Embarked':'nominal',
                         'Title':'nominal',
                        'Deck':'nominal', 
                        'Family_Size':'ordinal'}

把若干列标注为 `ordinal（有序）`、`nominal（无序类别）`或 `numeric（数值）`。这是元信息，不改变数据本身。

## imputing nan values 填补缺失值
缺失值情况统计：
| 列名        |   NaN个数 |   缺失比例(%) |
|:------------|----------:|--------------:|
| PassengerId |         0 |          0    |
| Survived    |         0 |          0    |
| Pclass      |         0 |          0    |
| Name        |         0 |          0    |
| Sex         |         0 |          0    |
| Age         |       177 |         19.87 |
| SibSp       |         0 |          0    |
| Parch       |         0 |          0    |
| Ticket      |         0 |          0    |
| Fare        |         0 |          0    |
| Cabin       |       687 |         77.1  |
| Embarked    |         2 |          0.22 |

| 列名        |   NaN个数 |   缺失比例(%) |
|:------------|----------:|--------------:|
| PassengerId |         0 |          0    |
| Pclass      |         0 |          0    |
| Name        |         0 |          0    |
| Sex         |         0 |          0    |
| Age         |        86 |         20.57 |
| SibSp       |         0 |          0    |
| Parch       |         0 |          0    |
| Ticket      |         0 |          0    |
| Fare        |         1 |          0.24 |
| Cabin       |       327 |         78.23 |
| Embarked    |         0 |          0    |

所以主要是 Age, Cabin, Embarked, Fare 有缺失值。
而 Cabin 缺失值过多，且可能本身缺失就有信息量（缺失可能代表没有船舱），所以不填补，在前面已经用'Ukown'代替。
### Fare 使用 Pclass 平均值填充
- 含义：用同舱位的平均票价填补 Fare 的缺失值。

        classmeans = df.pivot_table(values='Fare', index='Pclass', aggfunc='mean')['Fare']

- 使用 pivot_table 按 Pclass（舱位等级）计算每个 Pclass 下 Fare 的平均值，得到一个以 Pclass 为索引、平均票价为值的 Series（名为 classmeans）。

- 目的：用同舱位的平均票价来填补票价的缺失。

        df['Fare'] = df['Fare'].fillna(df['Pclass'].map(classmeans))

- 用 Pclass 映射 (map(classmeans)) 得到每行对应的舱位平均票价，然后把 Fare 中为 NaN 的位置以该映射值填充。

- 结果：缺失的 Fare 被相同 Pclass 的平均票价替代
### Age 使用平均值

        meanAge=np.mean(df.Age)
        df.Age=df.Age.fillna(meanAge)

### Embarked 使用众数填充

        modeEmbarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(modeEmbarked)
 
## Fare_Per_Person
含义：新增列 Fare_Per_Person，等于 Fare 除以 (Family_Size + 1)。+1 的用意是把乘客本人也算进人数（Family_Size 通常表示除本人外的随行人数），从而得到“每位乘客分摊到的票价”


        df['Fare_Per_Person'] = df['Fare']/(df['Family_Size']+1)

## Age*Class
- 含义：新增列 Age*Class，等于 Age 乘以 Pclass。结合年龄和舱位等级的信息，可能有助于捕捉某些模式（例如年轻人更可能选择低舱位）。

        df['Age*Class'] = df.Age * df.Pclass


## 增加新特征的数据类型标注
在之前的 data_type_dict 中加入新生成的两个特征的类型标注，表示它们是数值型（numeric）

        data_type_dict['Fare_Per_Person']='numeric'
        data_type_dict['Age*Class']='numeric'

## 数值类型变量进行分箱操作

| 方法        | 描述                                | 适用场景     |
| --------- | --------------------------------- | -------- |
| **等距分箱**  | 按相等区间划分，例如 `[0-10], [10-20], ...` | 数据分布较均匀时 |
| **等频分箱**  | 每个箱样本数相等                          | 数据分布偏斜时  |
| **自定义分箱** | 人工设定区间（如年龄段）                      | 结合业务知识   |


- 含义：对数值型变量进行分箱操作，把连续变量转换为类别变量，从而捕捉非线性关系。

        def discretise_numeric(train, test, data_type_dict, no_bins=10):
        N=len(train)
        M=len(test)
        test=test.rename(lambda x: x+N)
        joint_df = pd.concat([train, test])
        for column in data_type_dict:
            if data_type_dict[column]=='numeric':
                if not pd.api.types.is_numeric_dtype(joint_df[column]):
                    print(f"❗ Column {column} is marked numeric but isn't. Actual dtype: {joint_df[column].dtype}")
                joint_df[column] = pd.qcut(joint_df[column], no_bins, duplicates='drop')
                data_type_dict[column]='ordinal'
        train = joint_df.iloc[:N]
        test = joint_df.iloc[N:N+M]
        return train, test, data_type_dict




| 列名 | 训练集有效取值 | 
| --- | --- | 
| Pclass | 1, 2, 3 | 
| Age | (0.169, 21.0], (21.0, 28.0], (28.0, 30.0], (30.0, 39.0], (39.0, 80.0] | 
| Fare | (3.17, 7.879], (7.879, 11.5], (11.5, 22.358], (22.358, 46.06], (46.06, 512.329] | 
| Family_Size | 0, 1, 2, 3, 4, 5, 6, 7, 10 |
| Fare_Per_Person | (1.11, 7.228], (7.228, 7.896], (7.896, 12.0], (12.0, 27.35], (27.35, 512.329] |
| Age*Class | (0.509, 36.0], (36.0, 54.0], (54.0, 72.0], (72.0, 89.097], (89.097, 222.0] | 



## 最终特征
- `Pclass`: ordinal
- `Sex`: nominal
- `Age`: ordinal
- `Fare`: ordinal
- `Embarked`: nominal
- `Title`: nominal
- `Deck`: nominal
- `Family_Size`: ordinal
- `Fare_Per_Person`: ordinal
- `Age*Class`: ordinal
