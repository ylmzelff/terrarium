
import os

def disable_personal_assistant():
    """
    Disables PersonalAssistant imports in the codebase to allow running on Colab
    where the private CoLLAB submodule is missing.
    """
    print("Patching Terrarium for Colab/No-CoLLAB environment...")
    
    files_to_patch = {
        "envs/dcops/__init__.py": [
            ("from .personal_assistant import PersonalAssistantEnvironment", "# from .personal_assistant import PersonalAssistantEnvironment"),
            ("'PersonalAssistantEnvironment',", "# 'PersonalAssistantEnvironment',")
        ],
        "src/toolset_discovery.py": [
            ("from envs.dcops.personal_assistant.personal_assistant_tools import PersonalAssistantTools", "# from envs.dcops.personal_assistant.personal_assistant_tools import PersonalAssistantTools"),
            ("self.personal_assistant_tools = PersonalAssistantTools(blackboard_manager=None)", "# self.personal_assistant_tools = PersonalAssistantTools(blackboard_manager=None)"),
            ('"PersonalAssistantEnvironment": self.personal_assistant_tools,', '# "PersonalAssistantEnvironment": self.personal_assistant_tools,')
        ],
        "src/server.py": [
            ('elif environment_name == "PersonalAssistantEnvironment":', '# elif environment_name == "PersonalAssistantEnvironment":'),
            ('from envs.dcops.personal_assistant.personal_assistant_tools import PersonalAssistantTools', '# from envs.dcops.personal_assistant.personal_assistant_tools import PersonalAssistantTools'),
            ('environment_tools = PersonalAssistantTools(megaboard)', '# environment_tools = PersonalAssistantTools(megaboard)')
        ],
        "src/utils.py": [
            ("from envs.dcops.personal_assistant import PersonalAssistantEnvironment", "# from envs.dcops.personal_assistant import PersonalAssistantEnvironment"),
            ("PersonalAssistantEnvironment.__name__: PersonalAssistantEnvironment,", "# PersonalAssistantEnvironment.__name__: PersonalAssistantEnvironment,")
        ]
    }
    
    base_dir = os.getcwd()
    
    for rel_path, replacements in files_to_patch.items():
        file_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            for old, new in replacements:
                new_content = new_content.replace(old, new)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✓ Patched {rel_path}")
            else:
                print(f"- No changes needed for {rel_path}")
                
        except Exception as e:
            print(f"Error patching {rel_path}: {e}")

if __name__ == "__main__":
    disable_personal_assistant()
