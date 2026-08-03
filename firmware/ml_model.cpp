/**
 * Trek Guardian - Embedded ML Model Source
 * Generated: 2026-08-03 18:22:49 UTC
 * Distilled Decision Tree for early hypoxia warning.
 */

#include "ml_model.h"

String predictRisk(
  float altitude,
  float spo2,
  float heartRate,
  float spo2_ewma,
  float hr_ewma,
  float spo2_slope,
  float hr_slope
) {
    if (spo2_ewma <= 89.2277) {
        if (spo2 <= 88.2676) {
            if (spo2 <= 87.7889) {
                if (spo2_ewma <= 86.6773) {
                    if (altitude <= 2250.0000) {
                        return "CRITICAL";
                    } else {
                        return "CRITICAL";
                    }
                } else {
                    if (spo2_ewma <= 86.8551) {
                        return "CRITICAL";
                    } else {
                        return "CRITICAL";
                    }
                }
            } else {
                if (spo2_slope <= -0.0409) {
                    return "CRITICAL";
                } else {
                    return "CRITICAL";
                }
            }
        } else {
            if (hr_slope <= 0.0063) {
                if (hr_ewma <= 104.1666) {
                    if (spo2_slope <= -0.0451) {
                        return "HIGH";
                    } else {
                        return "CRITICAL";
                    }
                } else {
                    if (hr_slope <= -0.1964) {
                        return "CRITICAL";
                    } else {
                        return "CRITICAL";
                    }
                }
            } else {
                if (hr_ewma <= 148.3301) {
                    if (spo2 <= 88.8444) {
                        return "HIGH";
                    } else {
                        if (hr_ewma <= 85.8621) {
                            return "HIGH";
                        } else {
                            return "CRITICAL";
                        }
                    }
                } else {
                    return "CRITICAL";
                }
            }
        }
    } else {
        if (spo2_ewma <= 95.8273) {
            if (spo2_ewma <= 92.8559) {
                if (hr_ewma <= 114.4477) {
                    if (spo2_ewma <= 91.8421) {
                        if (spo2_slope <= 0.0708) {
                            if (spo2 <= 88.3360) {
                                return "CRITICAL";
                            } else {
                                return "HIGH";
                            }
                        } else {
                            if (altitude <= 3250.0000) {
                                return "CRITICAL";
                            } else {
                                return "HIGH";
                            }
                        }
                    } else {
                        if (hr_ewma <= 96.7641) {
                            if (spo2 <= 91.8500) {
                                return "HIGH";
                            } else {
                                return "HIGH";
                            }
                        } else {
                            if (spo2_ewma <= 92.2833) {
                                return "HIGH";
                            } else {
                                return "HIGH";
                            }
                        }
                    }
                } else {
                    if (spo2_ewma <= 91.4131) {
                        if (spo2_slope <= -0.0648) {
                            return "CRITICAL";
                        } else {
                            if (spo2_slope <= -0.0270) {
                                return "CRITICAL";
                            } else {
                                return "CRITICAL";
                            }
                        }
                    } else {
                        if (spo2 <= 91.5476) {
                            if (spo2_ewma <= 91.9052) {
                                return "HIGH";
                            } else {
                                return "HIGH";
                            }
                        } else {
                            if (spo2 <= 92.7616) {
                                return "CRITICAL";
                            } else {
                                return "HIGH";
                            }
                        }
                    }
                }
            } else {
                if (hr_ewma <= 59.5179) {
                    if (altitude <= 6250.0000) {
                        if (altitude <= 5750.0000) {
                            if (hr_ewma <= 54.0480) {
                                return "HIGH";
                            } else {
                                return "MODERATE";
                            }
                        } else {
                            if (heartRate <= 58.5000) {
                                return "HIGH";
                            } else {
                                return "HIGH";
                            }
                        }
                    } else {
                        if (altitude <= 7250.0000) {
                            if (spo2_slope <= 0.0591) {
                                return "MODERATE";
                            } else {
                                return "HIGH";
                            }
                        } else {
                            return "HIGH";
                        }
                    }
                } else {
                    if (altitude <= 4000.0000) {
                        if (hr_ewma <= 157.8465) {
                            if (hr_ewma <= 103.9052) {
                                return "CRITICAL";
                            } else {
                                return "CRITICAL";
                            }
                        } else {
                            return "HIGH";
                        }
                    } else {
                        if (spo2_slope <= -0.0469) {
                            if (altitude <= 6750.0000) {
                                return "MODERATE";
                            } else {
                                return "HIGH";
                            }
                        } else {
                            if (spo2_ewma <= 94.8450) {
                                return "MODERATE";
                            } else {
                                return "MODERATE";
                            }
                        }
                    }
                }
            }
        } else {
            if (spo2 <= 96.7500) {
                if (spo2_slope <= -0.0266) {
                    if (altitude <= 6250.0000) {
                        if (spo2 <= 95.7500) {
                            if (spo2 <= 95.2500) {
                                return "MODERATE";
                            } else {
                                return "MODERATE";
                            }
                        } else {
                            if (spo2_slope <= -0.0438) {
                                return "LOW";
                            } else {
                                return "LOW";
                            }
                        }
                    } else {
                        if (spo2_slope <= -0.0445) {
                            if (spo2 <= 95.8500) {
                                return "HIGH";
                            } else {
                                return "MODERATE";
                            }
                        } else {
                            if (spo2_ewma <= 96.0720) {
                                return "MODERATE";
                            } else {
                                return "MODERATE";
                            }
                        }
                    }
                } else {
                    if (spo2_slope <= 0.0926) {
                        if (hr_ewma <= 81.7793) {
                            if (altitude <= 6750.0000) {
                                return "LOW";
                            } else {
                                return "LOW";
                            }
                        } else {
                            if (hr_slope <= 0.3289) {
                                return "LOW";
                            } else {
                                return "MODERATE";
                            }
                        }
                    } else {
                        if (spo2_ewma <= 96.3845) {
                            if (spo2_ewma <= 96.0154) {
                                return "HIGH";
                            } else {
                                return "HIGH";
                            }
                        } else {
                            return "MODERATE";
                        }
                    }
                }
            } else {
                if (hr_ewma <= 123.1266) {
                    if (spo2_slope <= -0.0170) {
                        if (altitude <= 6750.0000) {
                            if (hr_slope <= 0.2882) {
                                return "LOW";
                            } else {
                                return "LOW";
                            }
                        } else {
                            if (spo2 <= 97.6500) {
                                return "MODERATE";
                            } else {
                                return "LOW";
                            }
                        }
                    } else {
                        if (spo2_ewma <= 97.6948) {
                            if (spo2_slope <= 0.0710) {
                                return "LOW";
                            } else {
                                return "MODERATE";
                            }
                        } else {
                            if (altitude <= 7250.0000) {
                                return "LOW";
                            } else {
                                return "LOW";
                            }
                        }
                    }
                } else {
                    return "HIGH";
                }
            }
        }
    }
}
