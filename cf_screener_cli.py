# cf_screener_cli.py
import numpy as np

# Try LiteRT first; fall back to tf.lite if not installed
try:
    from ai_edge_litert.interpreter import Interpreter
    USE_LITERT = True
except ImportError:
    import tensorflow as tf
    USE_LITERT = False

def get_input(prompt, expected_type="int", choices=None):
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                print("⚠️  Input cannot be empty. Please try again.")
                continue
            if expected_type == "int":
                val = int(value)
            elif expected_type == "float":
                val = float(value)
            else:
                val = value
            
            if choices and val not in choices:
                print(f"⚠️  Please enter one of: {choices}")
                continue
            return val
        except ValueError:
            print("⚠️  Invalid input. Please try again.")

def main():
    print("🩺 Cystic Fibrosis Risk Screener (for infants ≤24 months)")
    print("=" * 60)
    
    # Collect core inputs
    age = get_input("Age (months): ", "float")
    if age > 24:
         print("❌ This tool is only validated for infants ≤24 months. Please consult a specialist.")
         return
    elif age < 0:
        print("⚠️  Age must be ≥ 0.")
        return

    family_hx = get_input("Family history of CF? (0=No, 1=Yes): ", "int", [0, 1])
    salty_skin = get_input("Salty-tasting skin? (0=No, 1=Yes): ", "int", [0, 1])
    weight_pct = get_input("Weight percentile (0–100): ", "float")
    if not (0 <= weight_pct <= 100):
        print("⚠️  Weight percentile must be between 0 and 100.")
        return

    sweat_cl = get_input("Sweat chloride (mmol/L, enter -1 if unknown): ", "float")
    if sweat_cl < 0:
        sweat_cl = 25.0  # Default neutral value if unknown

    ethnicity = get_input("Ethnicity (0=Caucasian/Ashkenazi, 1=Other): ", "int", [0, 1])

    # Optional: Meconium ileus (only for age ≤ 2 months)
    meconium_ileus = 0
    if age <= 2:
        meconium_ileus = get_input("Meconium ileus at birth? (0/1): ", "int", [0, 1])

    # Set reasonable defaults for unasked features (based on low-risk profile)
    cough_type = 0
    resp_infections = 0
    wheezing = 0
    stool_char = 0  # 0 = normal

    # Derived features
    growth_faltering = 1 if weight_pct < 10 else 0
    resp_score = cough_type + resp_infections + wheezing
    nutr_score = (growth_faltering * 3) + (1 if stool_char != 0 else 0)

    # Prepare input array — MUST match training feature order!
    sample = np.array([[
        age,
        family_hx,
        ethnicity,
        salty_skin,
        cough_type,
        resp_infections,
        wheezing,
        weight_pct,
        growth_faltering,
        stool_char,
        meconium_ileus,
        sweat_cl,
        resp_score,
        nutr_score
    ]], dtype=np.float32)

    # Load model
    model_path = "cf_screening_model.tflite"
    try:
        if USE_LITERT:
            interpreter = Interpreter(model_path=model_path)
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        risk_score = float(output_data[0][0])

        # Display result
        print("\n" + "="*60)
        print(f"Risk Score: {risk_score:.4f}")
        if risk_score >= 0.5:
            print("🚨 HIGH RISK: Refer for sweat test and genetic confirmation!")
        else:
            print("✅ LOW RISK: Routine pediatric care.")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        print("Make sure 'cf_screening_model.tflite' exists in this folder.")
        if not USE_LITERT:
            print("💡 Consider installing LiteRT: pip install ai-edge-litert")

if __name__ == "__main__":
    main()