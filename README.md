# Air Passenger Satisfaction Prediction

This project is designed to predict the satisfaction of air passengers based on several features such as flight distance, onboard services, and delays. The goal is to classify passengers into two categories: satisfied or neutral/dissatisfied.

## Dataset
The dataset contains the following columns:
- **Age**: Age of the passenger.
- **Flight Distance**: The distance traveled by the passenger.
- **Inflight wifi service**: Rating of the inflight wifi service.
- **Seat comfort**: Rating of the seat comfort.
- **Inflight entertainment**: Rating of the inflight entertainment.
- **On-board service**: Rating of the on-board service.
- **Leg room service**: Rating of the leg room service.
- **Baggage handling**: Rating of baggage handling.
- **Checkin service**: Rating of the checkin service.
- **Inflight service**: Rating of the inflight service.
- **Cleanliness**: Rating of the cleanliness.
- **Departure Delay in Minutes**: Departure delay time.
- **Arrival Delay in Minutes**: Arrival delay time.
- **Class**: Class of travel (Eco, Eco Plus, Business).
- **Gender**: Gender of the passenger.
- **Type of Travel**: Whether the travel is personal or business.
- **Customer Type**: Type of customer (Loyal or Disloyal).

## Installation

To run this project, you will need the following Python packages:

- pandas
- scikit-learn
- numpy
- joblib
- lazypredict

You can install the required packages by running:

```bash
pip install -r requirements.txt
