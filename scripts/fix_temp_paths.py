from pathlib import Path

def fix_temp_paths():
    train_file = Path('src/models/train.py')
    
    if not train_file.exists():
        print(f"File not found: {train_file}")
        return
    
    content = train_file.read_text()
    
    # Replace /tmp/ with outputs/
    content = content.replace("'/tmp/", "'outputs/")
    content = content.replace('"/tmp/', '"outputs/')
    
    train_file.write_text(content)
    
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("✓ Fixed temp paths in train.py")
    print("✓ Created outputs directory")

if __name__ == "__main__":
    fix_temp_paths()