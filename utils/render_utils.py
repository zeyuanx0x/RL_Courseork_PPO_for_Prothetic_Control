def safe_render(env, render: bool = True):
    """Safely render environment, handling different rendering methods
    
    Args:
        env: Environment instance
        render: Whether to render
    """
    if not render:
        return
        
    try:
        # Try direct mj_render call (MyoSuite environments)
        env.mj_render()
    except AttributeError:
        try:
            # Try through unwrapped environment
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'mj_render'):
                env.unwrapped.mj_render()
            else:
                # Try standard render method
                env.render()
        except Exception as e:
            # Ignore rendering errors, don't affect main process
            pass

def setup_render(env, **kwargs):
    """Set up rendering parameters, handle rendering initialization uniformly
    
    Args:
        env: Environment instance
        **kwargs: Rendering parameters
    """
    try:
        # Try to set render mode
        if hasattr(env, 'render_mode'):
            env.render_mode = kwargs.get('render_mode', 'human')
    except Exception as e:
        # Ignore setup errors
        pass