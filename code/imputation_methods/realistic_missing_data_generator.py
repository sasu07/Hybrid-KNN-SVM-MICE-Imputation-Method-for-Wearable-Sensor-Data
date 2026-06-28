import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class RealisticMissingDataGenerator:
    """
    A class to generate realistic missing data scenarios for wearable sensor data.
    Implements six realistic scenarios across three missing rates.
    """
    
    def __init__(self, data_path, output_dir):
        """
        Initialize the generator with the dataset path and output directory.
        
        Parameters:
        -----------
        data_path : str
            Path to the complete dataset CSV file
        output_dir : str
            Directory to save the generated missing data scenarios
        """
        self.data = pd.read_csv(data_path)
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save a copy of the original data
        self.data.to_csv(os.path.join(output_dir, 'original_data.csv'), index=False)
        
        # Identify sensor groups based on column names
        self.sensor_groups = self._identify_sensor_groups()
        
        # Define missing rates
        self.missing_rates = [0.1, 0.2, 0.3]  # 10%, 20%, 30%
        
    def _identify_sensor_groups(self):
        """Identify sensor groups based on column names"""
        cols = list(self.data.columns)
        # HAR70+ schema: back_* and thigh_* channels, magnitudes, correlations.
        # Each "sensor group" mimics a physical sensor that can fail together.
        sensor_groups = {
            'accelerometer': [c for c in cols if c.startswith('back_x') or c.startswith('back_y') or c.startswith('back_z')],
            'gyroscope': [c for c in cols if c.startswith('thigh_x') or c.startswith('thigh_y') or c.startswith('thigh_z')],
            'orientation': [c for c in cols if 'corr_' in c],
            'magnitude': [c for c in cols if 'mag' in c],
            'angles': [],
            'hrm': [],
            'metadata': ['Activity', 'time', 'participant']
        }
        # Fallback for the original wearable dataset (keeps backward compatibility)
        if not any(sensor_groups[g] for g in ('accelerometer','gyroscope','magnitude')):
            sensor_groups = {
                'accelerometer': [c for c in cols if 'Accelerometer' in c],
                'gyroscope': [c for c in cols if 'Gyroscope' in c],
                'orientation': [c for c in cols if 'Orientation' in c],
                'magnitude': [c for c in cols if 'Magnitude' in c],
                'angles': ['Roll', 'Pitch', 'Yaw'],
                'hrm': ['hrm'],
                'metadata': ['Activity', 'time', 'participant']
            }
        return sensor_groups
    
    def _get_numerical_columns(self):
        """Get all numerical columns excluding metadata"""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # Remove metadata columns if they happen to be numerical
        for col in self.sensor_groups['metadata']:
            if col in numerical_cols:
                numerical_cols.remove(col)
        return numerical_cols
    
    def generate_all_scenarios(self):
        """Generate all missing data scenarios"""
        scenarios = []
        
        # Generate scenarios for each missing rate
        for rate in self.missing_rates:
            # MCAR scenarios
            scenarios.append(self.random_sensor_failures(rate))
            scenarios.append(self.temporary_connection_issues(rate))
            
            # MAR scenarios
            scenarios.append(self.activity_dependent_failures(rate))
            scenarios.append(self.battery_depletion_patterns(rate))
            
            # MNAR scenarios
            scenarios.append(self.value_dependent_failures(rate))
            scenarios.append(self.sensor_range_limitations(rate))
        
        # Create a summary of all scenarios
        self._create_summary(scenarios)
        
        return scenarios
    
    def random_sensor_failures(self, missing_rate):
        """
        Scenario 1: Random Sensor Failures (MCAR)
        Individual sensors randomly fail for short periods.
        """
        scenario_name = f"MCAR_RandomSensorFailures_{int(missing_rate*100)}pct"
        print(f"Generating {scenario_name}...")
        
        # Create a copy of the data
        data_copy = self.data.copy()
        numerical_cols = self._get_numerical_columns()
        
        # Calculate total cells to set as missing
        total_cells = len(data_copy) * len(numerical_cols)
        target_missing = int(total_cells * missing_rate)
        current_missing = 0
        
        while current_missing < target_missing:
            # Select a random sensor group
            sensor_group = np.random.choice(list(self.sensor_groups.keys()))
            if sensor_group == 'metadata':
                continue  # Skip metadata
                
            # Get columns for this sensor group
            columns = self.sensor_groups[sensor_group]
            columns = [col for col in columns if col in numerical_cols]
            
            if not columns:
                continue
                
            # Determine a random failure duration (5-20 consecutive measurements)
            failure_duration = np.random.randint(5, 21)
            
            # Select a random starting point
            start_idx = np.random.randint(0, len(data_copy) - failure_duration)
            
            # Set values to missing for the selected sensor group
            for col in columns:
                # Count non-NaN values that will be set to NaN
                non_nan_count = data_copy.loc[start_idx:start_idx+failure_duration-1, col].notna().sum()
                data_copy.loc[start_idx:start_idx+failure_duration-1, col] = np.nan
                current_missing += non_nan_count
                
            # Break if we've exceeded the target
            if current_missing >= target_missing:
                break
        
        # Save the scenario
        output_path = os.path.join(self.output_dir, f"{scenario_name}.csv")
        data_copy.to_csv(output_path, index=False)
        
        # Calculate actual missing rate
        actual_missing_rate = data_copy[numerical_cols].isna().sum().sum() / total_cells
        
        return {
            'name': scenario_name,
            'type': 'MCAR',
            'description': 'Random Sensor Failures',
            'target_rate': missing_rate,
            'actual_rate': actual_missing_rate,
            'missing_count': data_copy[numerical_cols].isna().sum().sum(),
            'path': output_path
        }
    
    def temporary_connection_issues(self, missing_rate):
        """
        Scenario 2: Temporary Connection Issues (MCAR)
        All sensors fail simultaneously for short periods.
        """
        scenario_name = f"MCAR_ConnectionIssues_{int(missing_rate*100)}pct"
        print(f"Generating {scenario_name}...")
        
        # Create a copy of the data
        data_copy = self.data.copy()
        numerical_cols = self._get_numerical_columns()
        
        # Calculate total cells to set as missing
        total_cells = len(data_copy) * len(numerical_cols)
        target_missing = int(total_cells * missing_rate)
        current_missing = 0
        
        while current_missing < target_missing:
            # Determine a random connection issue duration (3-15 consecutive measurements)
            issue_duration = np.random.randint(3, 16)
            
            # Select a random starting point
            start_idx = np.random.randint(0, len(data_copy) - issue_duration)
            
            # Set all sensor values to missing for the duration
            for col in numerical_cols:
                # Count non-NaN values that will be set to NaN
                non_nan_count = data_copy.loc[start_idx:start_idx+issue_duration-1, col].notna().sum()
                data_copy.loc[start_idx:start_idx+issue_duration-1, col] = np.nan
                current_missing += non_nan_count
                
            # Break if we've exceeded the target
            if current_missing >= target_missing:
                break
        
        # Save the scenario
        output_path = os.path.join(self.output_dir, f"{scenario_name}.csv")
        data_copy.to_csv(output_path, index=False)
        
        # Calculate actual missing rate
        actual_missing_rate = data_copy[numerical_cols].isna().sum().sum() / total_cells
        
        return {
            'name': scenario_name,
            'type': 'MCAR',
            'description': 'Temporary Connection Issues',
            'target_rate': missing_rate,
            'actual_rate': actual_missing_rate,
            'missing_count': data_copy[numerical_cols].isna().sum().sum(),
            'path': output_path
        }
    
    def activity_dependent_failures(self, missing_rate):
        """
        Scenario 3: Activity-Dependent Sensor Failures (MAR)
        Certain sensors are more likely to fail during specific activities.
        """
        scenario_name = f"MAR_ActivityDependent_{int(missing_rate*100)}pct"
        print(f"Generating {scenario_name}...")
        
        # Create a copy of the data
        data_copy = self.data.copy()
        numerical_cols = self._get_numerical_columns()
        
        # Calculate total cells to set as missing
        total_cells = len(data_copy) * len(numerical_cols)
        target_missing = int(total_cells * missing_rate)
        current_missing = 0
        
        # Define activity-sensor relationships (which sensors are more likely to fail during which activities)
        # This is a simplified example - in reality, these relationships would be based on domain knowledge
        activity_sensor_map = {
            # Original wearable dataset (Romanian labels)
            'Mers': self.sensor_groups['accelerometer'],
            'Alergare': self.sensor_groups['gyroscope'],
            'Urcat': self.sensor_groups['orientation'],
            'Coborat': self.sensor_groups['angles'],
            # HAR70+ labels
            'walking': self.sensor_groups['accelerometer'],
            'sitting': self.sensor_groups['gyroscope'],
            'standing': self.sensor_groups['orientation'],
            'lying': self.sensor_groups['magnitude'],
        }
        
        # Get unique activities
        activities = data_copy['Activity'].unique()
        
        # For each activity, apply a higher probability of missingness to the associated sensors
        for activity in activities:
            if activity in activity_sensor_map:
                # Get rows for this activity
                activity_mask = data_copy['Activity'] == activity
                activity_rows = data_copy[activity_mask].index
                
                # Get associated sensors
                associated_sensors = activity_sensor_map[activity]
                associated_sensors = [col for col in associated_sensors if col in numerical_cols]
                
                if not associated_sensors or len(activity_rows) == 0:
                    continue
                
                # Calculate how many values to set missing for this activity-sensor combination
                # We'll use a higher probability for these combinations
                probability = min(0.5, missing_rate * 3)  # Higher probability, but capped at 50%
                
                # Set values to missing
                for col in associated_sensors:
                    for idx in activity_rows:
                        if np.random.random() < probability and data_copy.loc[idx, col] == data_copy.loc[idx, col]:  # Check if not already NaN
                            data_copy.loc[idx, col] = np.nan
                            current_missing += 1
                            
                            # Break if we've exceeded the target
                            if current_missing >= target_missing:
                                break
                    
                    if current_missing >= target_missing:
                        break
                        
                if current_missing >= target_missing:
                    break
        
        # Save the scenario
        output_path = os.path.join(self.output_dir, f"{scenario_name}.csv")
        data_copy.to_csv(output_path, index=False)
        
        # Calculate actual missing rate
        actual_missing_rate = data_copy[numerical_cols].isna().sum().sum() / total_cells
        
        return {
            'name': scenario_name,
            'type': 'MAR',
            'description': 'Activity-Dependent Sensor Failures',
            'target_rate': missing_rate,
            'actual_rate': actual_missing_rate,
            'missing_count': data_copy[numerical_cols].isna().sum().sum(),
            'path': output_path
        }
    
    def battery_depletion_patterns(self, missing_rate):
        """
        Scenario 4: Battery Depletion Patterns (MAR)
        Sensors gradually fail as battery depletes over time.
        """
        scenario_name = f"MAR_BatteryDepletion_{int(missing_rate*100)}pct"
        print(f"Generating {scenario_name}...")
        
        # Create a copy of the data
        data_copy = self.data.copy()
        numerical_cols = self._get_numerical_columns()
        
        # Calculate total cells to set as missing
        total_cells = len(data_copy) * len(numerical_cols)
        target_missing = int(total_cells * missing_rate)
        
        # Define the order in which sensors fail as battery depletes
        # This is based on power consumption (hypothetical order)
        sensor_failure_order = [
            self.sensor_groups['hrm'],           # Heart rate monitor fails first (high power consumption)
            self.sensor_groups['gyroscope'],     # Then gyroscope
            self.sensor_groups['orientation'],   # Then orientation sensors
            self.sensor_groups['accelerometer']  # Accelerometer lasts longest
        ]
        
        # Flatten the list and keep only numerical columns
        sensor_failure_order = [col for group in sensor_failure_order for col in group if col in numerical_cols]
        
        # Calculate failure points along the time dimension
        # We'll simulate battery depletion by making sensors fail progressively from the end of the dataset
        n_rows = len(data_copy)
        
        # Calculate how many rows to affect to achieve the target missing rate
        # If we set all sensors in sensor_failure_order to missing for X rows,
        # we need to solve: X * len(sensor_failure_order) / total_cells = missing_rate
        rows_to_affect = int(target_missing / len(sensor_failure_order))
        rows_to_affect = min(rows_to_affect, n_rows)  # Ensure we don't exceed dataset size
        
        # Calculate failure points for each sensor group
        # Earlier sensors in the order will fail earlier (i.e., affect more rows)
        failure_points = {}
        for i, sensor in enumerate(sensor_failure_order):
            # Calculate how many rows this sensor will be missing
            # The first sensor in the order will be missing for all affected rows
            # The last sensor will be missing for fewer rows
            sensor_affected_rows = int(rows_to_affect * (1 - i/len(sensor_failure_order)))
            failure_points[sensor] = max(0, n_rows - sensor_affected_rows)
        
        # Set values to missing based on failure points
        current_missing = 0
        for sensor, failure_point in failure_points.items():
            # Set all values from failure point onwards to missing
            non_nan_count = data_copy.loc[failure_point:, sensor].notna().sum()
            data_copy.loc[failure_point:, sensor] = np.nan
            current_missing += non_nan_count
        
        # Save the scenario
        output_path = os.path.join(self.output_dir, f"{scenario_name}.csv")
        data_copy.to_csv(output_path, index=False)
        
        # Calculate actual missing rate
        actual_missing_rate = data_copy[numerical_cols].isna().sum().sum() / total_cells
        
        return {
            'name': scenario_name,
            'type': 'MAR',
            'description': 'Battery Depletion Patterns',
            'target_rate': missing_rate,
            'actual_rate': actual_missing_rate,
            'missing_count': data_copy[numerical_cols].isna().sum().sum(),
            'path': output_path
        }
    
    def value_dependent_failures(self, missing_rate):
        """
        Scenario 5: Value-Dependent Sensor Failures (MNAR)
        Sensors are more likely to fail when they produce extreme values.
        """
        scenario_name = f"MNAR_ValueDependent_{int(missing_rate*100)}pct"
        print(f"Generating {scenario_name}...")
        
        # Create a copy of the data
        data_copy = self.data.copy()
        numerical_cols = self._get_numerical_columns()
        
        # Calculate total cells to set as missing
        total_cells = len(data_copy) * len(numerical_cols)
        target_missing = int(total_cells * missing_rate)
        current_missing = 0
        
        # For each numerical column, identify extreme values and set some to missing
        for col in numerical_cols:
            # Calculate absolute values to identify extremes in both directions
            abs_values = data_copy[col].abs()
            
            # Identify extreme values (top 20% of absolute values)
            threshold = abs_values.quantile(0.8)
            extreme_mask = abs_values > threshold
            
            # Calculate how many extreme values to set as missing
            # We want to set a higher proportion of extreme values to missing
            extreme_count = extreme_mask.sum()
            if extreme_count == 0:
                continue
                
            # Set a proportion of extreme values to missing
            # The proportion is higher for more extreme values
            for idx in data_copy[extreme_mask].index:
                # Calculate probability based on how extreme the value is
                # More extreme values have higher probability of being missing
                extremeness = (abs_values[idx] - threshold) / (abs_values.max() - threshold)
                probability = min(0.8, missing_rate * 3 * extremeness)  # Cap at 80%
                
                if np.random.random() < probability and data_copy.loc[idx, col] == data_copy.loc[idx, col]:  # Check if not already NaN
                    data_copy.loc[idx, col] = np.nan
                    current_missing += 1
                    
                    # Break if we've exceeded the target
                    if current_missing >= target_missing:
                        break
            
            if current_missing >= target_missing:
                break
        
        # Save the scenario
        output_path = os.path.join(self.output_dir, f"{scenario_name}.csv")
        data_copy.to_csv(output_path, index=False)
        
        # Calculate actual missing rate
        actual_missing_rate = data_copy[numerical_cols].isna().sum().sum() / total_cells
        
        return {
            'name': scenario_name,
            'type': 'MNAR',
            'description': 'Value-Dependent Sensor Failures',
            'target_rate': missing_rate,
            'actual_rate': actual_missing_rate,
            'missing_count': data_copy[numerical_cols].isna().sum().sum(),
            'path': output_path
        }
    
    def sensor_range_limitations(self, missing_rate):
        """
        Scenario 6: Sensor Range Limitations (MNAR)
        Sensors fail to record values outside their operating range.
        """
        scenario_name = f"MNAR_RangeLimitation_{int(missing_rate*100)}pct"
        print(f"Generating {scenario_name}...")
        
        # Create a copy of the data
        data_copy = self.data.copy()
        numerical_cols = self._get_numerical_columns()
        
        # Calculate total cells to set as missing
        total_cells = len(data_copy) * len(numerical_cols)
        target_missing = int(total_cells * missing_rate)
        current_missing = 0
        
        # Define sensor range limitations
        # For each sensor, we'll define upper and lower limits based on percentiles
        # Values outside these limits will be set to missing
        sensor_ranges = {}
        for col in numerical_cols:
            # Define limits based on percentiles
            # The exact percentiles depend on the target missing rate
            # Higher missing rates = narrower ranges
            lower_percentile = missing_rate / 2
            upper_percentile = 1 - (missing_rate / 2)
            
            lower_limit = data_copy[col].quantile(lower_percentile)
            upper_limit = data_copy[col].quantile(upper_percentile)
            
            sensor_ranges[col] = (lower_limit, upper_limit)
        
        # Set values outside ranges to missing
        for col, (lower_limit, upper_limit) in sensor_ranges.items():
            # Identify values outside the range
            outside_range = (data_copy[col] < lower_limit) | (data_copy[col] > upper_limit)
            
            # Set these values to missing
            non_nan_count = data_copy.loc[outside_range, col].notna().sum()
            data_copy.loc[outside_range, col] = np.nan
            current_missing += non_nan_count
            
            # Break if we've exceeded the target
            if current_missing >= target_missing:
                break
        
        # Save the scenario
        output_path = os.path.join(self.output_dir, f"{scenario_name}.csv")
        data_copy.to_csv(output_path, index=False)
        
        # Calculate actual missing rate
        actual_missing_rate = data_copy[numerical_cols].isna().sum().sum() / total_cells
        
        return {
            'name': scenario_name,
            'type': 'MNAR',
            'description': 'Sensor Range Limitations',
            'target_rate': missing_rate,
            'actual_rate': actual_missing_rate,
            'missing_count': data_copy[numerical_cols].isna().sum().sum(),
            'path': output_path
        }
    
    def _create_summary(self, scenarios):
        """Create a summary of all scenarios"""
        summary_path = os.path.join(self.output_dir, 'scenarios_summary.csv')
        
        # Create a summary dataframe
        summary = pd.DataFrame([
            {
                'Scenario': s['name'],
                'Type': s['type'],
                'Description': s['description'],
                'Target_Missing_Rate': s['target_rate'],
                'Actual_Missing_Rate': s['actual_rate'],
                'Missing_Count': s['missing_count'],
                'File_Path': s['path']
            }
            for s in scenarios
        ])
        
        # Save the summary
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
        # Also create a detailed summary of missing percentages by column
        detailed_summary = pd.DataFrame(index=self._get_numerical_columns())
        
        for s in scenarios:
            # Load the scenario data
            scenario_data = pd.read_csv(s['path'])
            
            # Calculate missing percentage for each column
            missing_pct = scenario_data[self._get_numerical_columns()].isna().mean() * 100
            
            # Add to the detailed summary
            detailed_summary[s['name']] = missing_pct
        
        # Save the detailed summary
        detailed_summary_path = os.path.join(self.output_dir, 'missing_percentages_by_column.csv')
        detailed_summary.to_csv(detailed_summary_path)
        print(f"Detailed missing percentages by column saved to {detailed_summary_path}")

# Main execution
if __name__ == "__main__":
    # Set paths
    data_path = 'original_data.csv'
    output_dir = 'research_analysis/missing_data_scenarios'
    
    # Create generator
    generator = RealisticMissingDataGenerator(data_path, output_dir)
    
    # Generate all scenarios
    scenarios = generator.generate_all_scenarios()
    
    print(f"Generated {len(scenarios)} missing data scenarios.")
