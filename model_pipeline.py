import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2
from lazypredict.Supervised import LazyClassifier

#
train = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\data_machine\data_set\train.csv")
test = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\data_machine\data_set\test.csv")
#
file=ProfileReport(train,title="airline",explorative=True)
file.to_file("airline_passenger.html")
#
x_train = train.drop(["Unnamed: 0", "id", "satisfaction"], axis=1)
y_train = train["satisfaction"]
x_test = test.drop(["Unnamed: 0", "id", "satisfaction"], axis=1)
y_test = test["satisfaction"]
#
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
#
num_feature = ["Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink",
               "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service",
               "Cleanliness", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
ord_features = ["Class"]
cate_feature = ["Gender", "Type of Travel", "Customer Type"]
ordinal_order = [["Eco", "Eco Plus", "Business"]]
#
print("Giá trị duy nhất trong cột Class:", x_train["Class"].unique())
#
num_transform = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="median")),
    ("Scaler", StandardScaler())
])
ord_transform = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OrdinalEncoder(categories=ordinal_order))
])
cat_transform = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num_prepro", num_transform, num_feature),
    ("cat_prepro", cat_transform, cate_feature),
    ("ord_prepro", ord_transform, ord_features)
])
#
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=42)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
#
print("\nKết quả LazyPredict:\n", models)
top_models = models.nlargest(5, 'F1 Score')
print("\nTop 5 mô hình:\n", top_models)
#
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("to_positive", MinMaxScaler()),  # Đảm bảo dữ liệu không âm cho chi2
    ("feature_selection", SelectPercentile(score_func=chi2, percentile=50)),  # Giữ 50% đặc trưng tốt nhất
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])
#
param_dist = {
    "classifier__n_estimators": [100, 200, 300, 400, 500],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ['sqrt', 'log2']
}
#
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1_weighted',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)
#
random_search.fit(x_train, y_train)
#
print("\nSiêu tham số tốt nhất:", random_search.best_params_)
print("F1-score tốt nhất (cross-validation):", random_search.best_score_)
#
best_model = random_search.best_estimator_
y_pred = best_model.predict(x_test)
print("\nBáo cáo phân loại trên tập test:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
#
import joblib
joblib.dump(best_model, "best_model.pkl")
print("Mô hình đã được lưu vào 'best_model.pkl'")
