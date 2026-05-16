const AGE_GROUPS = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'];
const GENDERS = ['Male', 'Female', 'Non-binary', 'Prefer not to say'];

export function DemographicsModal({ onConfirm }) {
  function handleSubmit(e) {
    e.preventDefault();
    const data = new FormData(e.target);
    onConfirm({
      ageGroup: data.get('age_group'),
      gender:   data.get('gender'),
    });
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2 className="modal-title">Before we start</h2>
        <p className="modal-sub">
          This helps us build a representative panel. Takes 5 seconds.
        </p>

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="field">
            <label className="field-label">Age group</label>
            <div className="radio-group">
              {AGE_GROUPS.map((g) => (
                <label key={g} className="radio-option">
                  <input type="radio" name="age_group" value={g} required />
                  <span>{g}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label className="field-label">Gender</label>
            <div className="radio-group">
              {GENDERS.map((g) => (
                <label key={g} className="radio-option">
                  <input type="radio" name="gender" value={g} required />
                  <span>{g}</span>
                </label>
              ))}
            </div>
          </div>

          <button type="submit" className="btn btn-primary btn-full">
            Start watching →
          </button>
        </form>
      </div>
    </div>
  );
}
