# OptionsVisualizer

Simple Python Web App for viewing simulated options greeks and how they would pay off.
![image](https://github.com/user-attachments/assets/8fe41d1f-7b8a-4b65-82d1-cec303163e60)

## Features

- Calculate option prices using the Black-Scholes model
- Visualize option payoff diagrams for different strategies
- Interactive graphs for option Greeks (Delta, Gamma, Theta, Vega)
- Support for multiple option strategies:
  - Long/Short Calls and Puts
  - Covered Calls
  - Protective Puts
  - Bull Call Spreads
  - Bear Put Spreads
  - Iron Condors

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/OptionsVisualizer.git
cd OptionsVisualizer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:8050/`

3. Enter option parameters:
   - Stock price
   - Strike price
   - Volatility
   - Days to expiration
   - Risk-free rate
   - Option type (Call/Put)
   - Strategy

4. Click "Calculate" to view the option payoff and Greeks visualizations

## Technologies Used

- [Dash](https://dash.plotly.com/) - Interactive web application framework
- [Plotly](https://plotly.com/) - Interactive graphing library
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) - Bootstrap components for Dash

## License

This project is licensed under the MIT License - No particular reason I did this I just always use MIT license. 
