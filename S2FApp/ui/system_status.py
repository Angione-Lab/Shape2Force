"""System status UI component (CPU/memory)."""
import streamlit as st

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _get_container_memory():
    """
    Read memory from cgroups when running in a container (Docker, HF Spaces).
    psutil reports host memory in containers, which can be misleading (e.g. 128 GB vs 16 GB limit).
    Returns (used_bytes, total_bytes) or None to fall back to psutil.
    """
    try:
        # cgroup v2 (modern Docker, HF Spaces)
        for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/self"):
            try:
                with open(f"{base}/memory.max", "r") as f:
                    max_val = f.read().strip()
                if max_val == "max":
                    return None  # No limit, use psutil
                total = int(max_val)
                with open(f"{base}/memory.current", "r") as f:
                    used = int(f.read().strip())
                return (used, total)
            except (FileNotFoundError, ValueError):
                continue
        # cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                total = int(f.read().strip())
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r") as f:
                used = int(f.read().strip())
            if total > 2**50:  # Often 9223372036854771712 when unlimited
                return None
            return (used, total)
        except (FileNotFoundError, ValueError):
            pass
    except Exception:
        pass
    return None


def render_system_status():
    """Render CPU/memory status in the sidebar (always visible)."""
    if not HAS_PSUTIL:
        return
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        container_mem = _get_container_memory()
        if container_mem is not None:
            used_bytes, total_bytes = container_mem
            mem_used_gb = used_bytes / (1024**3)
            mem_total_gb = total_bytes / (1024**3)
            mem_pct = 100 * used_bytes / total_bytes if total_bytes > 0 else 0
        else:
            mem = psutil.virtual_memory()
            mem_used_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            mem_pct = mem.percent
        st.sidebar.markdown(
            f"""
            <div class="system-status">
                <span class="status-dot"></span>
                <span><strong>System</strong>&ensp;CPU {cpu:.0f}%&ensp;·&ensp;Mem {mem_pct:.0f}% ({mem_used_gb:.1f}/{mem_total_gb:.1f} GB)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass
